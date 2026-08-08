#include "blas_traits.cuh"
#include "engine_internal.cuh"
#include "huber_kernels.cuh"
#include "objective.cuh"
#include "workspace.cuh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace rh_cuda::engine {

template <typename T>
bool prefer_syrkx_for_gram(int rows, int parameters) noexcept {
    /*
     * SYRKX saves roughly half of the matrix multiplication work, but it also
     * needs a separate p-by-p mirror kernel and cuBLAS selects less efficient
     * kernels for narrow output matrices.  The crossover is dtype-dependent:
     * the standard shape sweep on an RTX 5070 Ti puts it above p=90 for both
     * types, while p=256 wins decisively.  Keep GEMM for narrow and short
     * batches; this also avoids paying the mirror cost when p^2 dominates the
     * useful row work.
     */
    constexpr int minimum_parameters = std::is_same_v<T, float> ? 192 : 128;
    return parameters >= minimum_parameters && rows >= parameters;
}

template <typename T>
void compute_weighted_gram(RhCudaEngine* engine, int rows, int parameters) {
    const T one = static_cast<T>(1);
    const T zero = static_cast<T>(0);
    if (prefer_syrkx_for_gram<T>(rows, parameters)) {
        check_cublas(
            Blas<T>::syrkx(
                engine->cublas,
                parameters,
                rows,
                &one,
                typed<T>(engine->d_design),
                parameters,
                typed<T>(engine->d_weighted_design),
                parameters,
                &zero,
                typed<T>(engine->d_gram),
                parameters
            ),
            "compute weighted Gram matrix with SYRKX"
        );
        check_cuda(
            rh_cuda::launch_mirror_lower_triangle(
                typed<T>(engine->d_gram), parameters, engine->stream
            ),
            "mirror weighted Gram matrix"
        );
        return;
    }
    check_cublas(
        Blas<T>::gemm(
            engine->cublas,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            parameters,
            parameters,
            rows,
            &one,
            typed<T>(engine->d_design),
            parameters,
            typed<T>(engine->d_weighted_design),
            parameters,
            &zero,
            typed<T>(engine->d_gram),
            parameters
        ),
        "compute weighted Gram matrix with GEMM"
    );
}





template <typename T>
void compute_residual(RhCudaEngine* engine, int rows, const T* beta) {
    const T negative_one = static_cast<T>(-1);
    const T one = static_cast<T>(1);
    const int parameters = static_cast<int>(engine->n_parameters);
    check_cublas(
        Blas<T>::copy(engine->cublas, rows, typed<T>(engine->d_y), typed<T>(engine->d_residual)),
        "copy y into residual"
    );
    /*
     * X_design arrives C-row-major (rows, parameters).  Its byte layout is a
     * column-major (parameters, rows) X^T, so GEMV with op=T computes X beta
     * without a transpose or an extra device allocation.
     */
    check_cublas(
        Blas<T>::gemv(
            engine->cublas,
            CUBLAS_OP_T,
            parameters,
            rows,
            &negative_one,
            typed<T>(engine->d_design),
            parameters,
            beta,
            &one,
            typed<T>(engine->d_residual)
        ),
        "compute residual"
    );
}


template <typename T>
void enqueue_smooth_objective(
    RhCudaEngine* engine,
    int rows,
    const T* beta,
    const T* weights,
    T tau,
    const T* previous_beta = nullptr
) {
    const int parameters = static_cast<int>(engine->n_parameters);
    compute_residual<T>(engine, rows, beta);
    check_cuda(
        rh_cuda::launch_huber_loss(
            typed<T>(engine->d_residual), weights, typed<T>(engine->d_loss), rows, tau, engine->stream
        ),
        "launch Huber loss"
    );
    T* reduction_results = typed<T>(engine->d_reduction_results);
    check_cublas(
        Blas<T>::asum(
            engine->cublas_reduction,
            rows,
            typed<T>(engine->d_loss),
            reduction_results
        ),
        "reduce Huber loss"
    );

    check_cuda(
        rh_cuda::launch_subtract(
            beta, typed<T>(engine->d_coefficients), typed<T>(engine->d_delta), parameters, engine->stream
        ),
        "form historical coefficient delta"
    );
    const T one = static_cast<T>(1);
    const T zero = static_cast<T>(0);
    check_cublas(
        Blas<T>::gemv(
            engine->cublas,
            CUBLAS_OP_N,
            parameters,
            parameters,
            &one,
            typed<T>(engine->d_information),
            parameters,
            typed<T>(engine->d_delta),
            &zero,
            typed<T>(engine->d_history_vector)
        ),
        "compute historical objective term"
    );
    check_cublas(
        Blas<T>::dot(
            engine->cublas_reduction,
            parameters,
            typed<T>(engine->d_delta),
            typed<T>(engine->d_history_vector),
            reduction_results + 1
        ),
        "reduce historical objective term"
    );
    if (previous_beta != nullptr) {
        /*
         * Candidate acceptance and Newton convergence are both host-side
         * decisions.  Queue the two convergence reductions behind the
         * objective work so all four scalars cross PCIe in one transfer and
         * require only one stream synchronization.  Rejected backtracking
         * candidates simply discard these two inexpensive reductions.
         */
        check_cuda(
            rh_cuda::launch_subtract(
                beta,
                previous_beta,
                typed<T>(engine->d_delta),
                parameters,
                engine->stream
            ),
            "form Newton coefficient difference"
        );
        check_cublas(
            Blas<T>::nrm2(
                engine->cublas_reduction,
                parameters,
                typed<T>(engine->d_delta),
                reduction_results + 2
            ),
            "reduce Newton coefficient difference"
        );
        check_cublas(
            Blas<T>::nrm2(
                engine->cublas_reduction,
                parameters,
                beta,
                reduction_results + 3
            ),
            "reduce Newton coefficient norm"
        );
    }
    T* host_results = typed<T>(engine->h_reduction_results);
    const size_t result_count = previous_beta == nullptr ? 2 : 4;
    check_cuda(
        cudaMemcpyAsync(
            host_results,
            reduction_results,
            result_count * sizeof(T),
            cudaMemcpyDeviceToHost,
            engine->stream
        ),
        "read objective reductions"
    );
}

template <typename T>
ObjectiveResult finish_smooth_objective(
    RhCudaEngine* engine,
    double n_total,
    bool has_previous_beta
) {
    check_cuda(cudaStreamSynchronize(engine->stream), "wait for objective reductions");
    T* host_results = typed<T>(engine->h_reduction_results);
    ObjectiveResult result;
    result.objective = (
        static_cast<double>(host_results[0]) + 0.5 * static_cast<double>(host_results[1])
    ) / n_total;
    if (has_previous_beta) {
        result.difference_norm = static_cast<double>(host_results[2]);
        result.beta_norm = static_cast<double>(host_results[3]);
    }
    return result;
}

template <typename T>
ObjectiveResult smooth_objective(
    RhCudaEngine* engine,
    int rows,
    const T* beta,
    const T* weights,
    T tau,
    double n_total,
    const T* previous_beta
) {
    enqueue_smooth_objective<T>(engine, rows, beta, weights, tau, previous_beta);
    return finish_smooth_objective<T>(engine, n_total, previous_beta != nullptr);
}

/*
 * One update owns one candidate-objective graph. Captured pointers therefore
 * cannot outlive a borrowed DLPack producer, and shape/config changes always
 * recapture. Capture is deliberately best-effort: unsupported cuBLAS/CUDA
 * combinations clear the capture and execute the strict stream path.
 */

CandidateObjectiveGraph::CandidateObjectiveGraph(RhCudaEngine* engine)
    : engine_(engine),
      enabled_((engine->enabled_flags & RH_CUDA_ENGINE_FLAG_CUDA_GRAPHS) != 0) {}

CandidateObjectiveGraph::~CandidateObjectiveGraph() noexcept {
    if (execution_ != nullptr) {
        cudaGraphExecDestroy(execution_);
    }
    if (graph_ != nullptr) {
        cudaGraphDestroy(graph_);
    }
}

template <typename T>
bool CandidateObjectiveGraph::capture(
    int rows,
    const T* beta,
    const T* weights,
    T tau,
    const T* previous_beta
)
{
    const cudaError_t begin = cudaStreamBeginCapture(
        engine_->stream, cudaStreamCaptureModeThreadLocal
    );
    if (begin != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    try {
        enqueue_smooth_objective<T>(engine_, rows, beta, weights, tau, previous_beta);
    } catch (...) {
        cudaGraph_t abandoned = nullptr;
        cudaStreamEndCapture(engine_->stream, &abandoned);
        if (abandoned != nullptr) {
            cudaGraphDestroy(abandoned);
        }
        cudaGetLastError();
        return false;
    }
    const cudaError_t end = cudaStreamEndCapture(engine_->stream, &graph_);
    if (end != cudaSuccess || graph_ == nullptr) {
        if (graph_ != nullptr) {
            cudaGraphDestroy(graph_);
            graph_ = nullptr;
        }
        cudaGetLastError();
        return false;
    }
    const cudaError_t instantiate = cudaGraphInstantiate(
        &execution_, graph_, nullptr, nullptr, 0
    );
    if (instantiate != cudaSuccess || execution_ == nullptr) {
        cudaGetLastError();
        return false;
    }
    ++engine_->graph_captures;
    return true;
}

template <typename T>
ObjectiveResult CandidateObjectiveGraph::evaluate(
    int rows,
    const T* beta,
    const T* weights,
    T tau,
    double n_total,
    const T* previous_beta
)
{
    if (!enabled_) {
        return smooth_objective<T>(
            engine_, rows, beta, weights, tau, n_total, previous_beta
        );
    }
    if (execution_ == nullptr && !capture<T>(rows, beta, weights, tau, previous_beta)) {
        ++engine_->graph_fallbacks;
        engine_->enabled_flags &= ~RH_CUDA_ENGINE_FLAG_CUDA_GRAPHS;
        enabled_ = false;
        return smooth_objective<T>(
            engine_, rows, beta, weights, tau, n_total, previous_beta
        );
    }
    check_cuda(cudaGraphLaunch(execution_, engine_->stream), "launch candidate objective graph");
    if (launched_once_) {
        ++engine_->graph_replays;
    } else {
        launched_once_ = true;
    }
    return finish_smooth_objective<T>(engine_, n_total, true);
}

template <typename T>
void compute_gradient_hessian(
    RhCudaEngine* engine,
    int rows,
    const T* beta,
    const T* weights,
    T tau,
    T bandwidth,
    double n_total,
    T ridge
) {
    const int parameters = static_cast<int>(engine->n_parameters);
    /*
     * solve_unpenalized evaluates the objective for the current trial before
     * every gradient/Hessian evaluation.  That objective leaves the matching
     * residual resident in d_residual, so recomputing y - X beta here would
     * duplicate a full-vector copy and GEMV on every Newton iteration.
     */
    check_cuda(
        rh_cuda::launch_residual_score_curvature(
            typed<T>(engine->d_residual),
            typed<T>(engine->d_residual),
            typed<T>(engine->d_score),
            typed<T>(engine->d_curvature),
            rows,
            tau,
            bandwidth,
            engine->stream
        ),
        "launch residual, Huber score, and curvature"
    );
    if (weights != nullptr) {
        check_cuda(
            rh_cuda::launch_weight_score(typed<T>(engine->d_score), weights, rows, engine->stream),
            "apply sample weight to score"
        );
    }

    const T negative_inv_total = static_cast<T>(-1.0 / n_total);
    const T positive_inv_total = static_cast<T>(1.0 / n_total);
    const T one = static_cast<T>(1);
    const T zero = static_cast<T>(0);
    check_cublas(
        Blas<T>::gemv(
            engine->cublas,
            CUBLAS_OP_N,
            parameters,
            rows,
            &negative_inv_total,
            typed<T>(engine->d_design),
            parameters,
            typed<T>(engine->d_score),
            &zero,
            typed<T>(engine->d_gradient)
        ),
        "compute current gradient"
    );
    check_cuda(
        rh_cuda::launch_subtract(
            beta, typed<T>(engine->d_coefficients), typed<T>(engine->d_delta), parameters, engine->stream
        ),
        "form gradient coefficient delta"
    );
    check_cublas(
        Blas<T>::gemv(
            engine->cublas,
            CUBLAS_OP_N,
            parameters,
            parameters,
            &positive_inv_total,
            typed<T>(engine->d_information),
            parameters,
            typed<T>(engine->d_delta),
            &one,
            typed<T>(engine->d_gradient)
        ),
        "add historical gradient"
    );

    check_cuda(
        rh_cuda::launch_weight_design(
            typed<T>(engine->d_design),
            typed<T>(engine->d_curvature),
            weights,
            typed<T>(engine->d_weighted_design),
            rows,
            parameters,
            engine->stream
        ),
        "form weighted design"
    );
    compute_weighted_gram<T>(engine, rows, parameters);
    const int64_t square = engine->n_parameters * engine->n_parameters;
    check_cuda(
        rh_cuda::launch_add_matrix(
            typed<T>(engine->d_gram),
            typed<T>(engine->d_information),
            typed<T>(engine->d_hessian),
            square,
            engine->stream
        ),
        "add historical information to Hessian"
    );
    check_cuda(
        rh_cuda::launch_scale_and_add_identity(
            typed<T>(engine->d_hessian),
            parameters,
            positive_inv_total,
            ridge,
            engine->stream
        ),
        "scale Hessian and add ridge"
    );
}
template <typename T>
void final_information(
    RhCudaEngine* engine,
    int rows,
    const T* weights,
    T tau,
    T bandwidth,
    bool residual_is_current
) {
    const int parameters = static_cast<int>(engine->n_parameters);
    if (!residual_is_current) {
        compute_residual<T>(engine, rows, typed<T>(engine->d_trial_beta));
    }
    check_cuda(
        rh_cuda::launch_residual_score_curvature(
            typed<T>(engine->d_residual),
            typed<T>(engine->d_residual),
            typed<T>(engine->d_score),
            typed<T>(engine->d_curvature),
            rows,
            tau,
            bandwidth,
            engine->stream
        ),
        "launch final curvature"
    );
    check_cuda(
        rh_cuda::launch_weight_design(
            typed<T>(engine->d_design),
            typed<T>(engine->d_curvature),
            weights,
            typed<T>(engine->d_weighted_design),
            rows,
            parameters,
            engine->stream
        ),
        "form final weighted design"
    );
    compute_weighted_gram<T>(engine, rows, parameters);
    check_cuda(
        rh_cuda::launch_add_matrix(
            typed<T>(engine->d_information),
            typed<T>(engine->d_gram),
            typed<T>(engine->d_information_next),
            engine->n_parameters * engine->n_parameters,
            engine->stream
        ),
        "form next renewable information"
    );
}

// Explicit instantiation: the engine is only ever float or double, and a
// missing pair fails the link instead of silently duplicating a definition.
template ObjectiveResult smooth_objective<float>(
    RhCudaEngine*, int, const float*, const float*, float, double, const float*
);
template ObjectiveResult smooth_objective<double>(
    RhCudaEngine*, int, const double*, const double*, double, double, const double*
);
template void compute_gradient_hessian<float>(
    RhCudaEngine*, int, const float*, const float*, float, float, double, float
);
template void compute_gradient_hessian<double>(
    RhCudaEngine*, int, const double*, const double*, double, double, double, double
);
template void final_information<float>(
    RhCudaEngine*, int, const float*, float, float, bool
);
template void final_information<double>(
    RhCudaEngine*, int, const double*, double, double, bool
);
template ObjectiveResult CandidateObjectiveGraph::evaluate<float>(
    int, const float*, const float*, float, double, const float*
);
template ObjectiveResult CandidateObjectiveGraph::evaluate<double>(
    int, const double*, const double*, double, double, const double*
);

}  // namespace rh_cuda::engine
