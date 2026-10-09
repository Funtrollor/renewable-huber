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


/*
 * One update owns one evaluator, so a captured round's pointers cannot outlive
 * a borrowed DLPack producer, and shape or configuration changes always
 * recapture.  The round's width, form and step scalars travel through a
 * pinned staging buffer that the round itself copies to the device, which is
 * what lets one graph serve every round of the update.
 */

CandidateRoundEvaluator::CandidateRoundEvaluator(RhCudaEngine* engine)
    : engine_(engine),
      graphs_((engine->enabled_flags & RH_CUDA_ENGINE_FLAG_CUDA_GRAPHS) != 0) {}

CandidateRoundEvaluator::~CandidateRoundEvaluator() noexcept {
    if (execution_ != nullptr) {
        cudaGraphExecDestroy(execution_);
    }
    if (graph_ != nullptr) {
        cudaGraphDestroy(graph_);
    }
}

template <typename T>
const T* CandidateRoundEvaluator::candidate(int k) const {
    return typed<T>(engine_->d_round_candidates) + static_cast<int64_t>(k) * engine_->n_parameters;
}

template <typename T>
const T* CandidateRoundEvaluator::residual(int rows, int k) const {
    return typed<T>(engine_->d_round_residuals) + static_cast<int64_t>(k) * rows;
}

template <typename T>
void CandidateRoundEvaluator::enqueue(
    int rows,
    const T* weights,
    T tau,
    const ObjectiveTerms<T>& terms
) {
    check_cuda(
        cudaMemcpyAsync(
            engine_->d_round_parameters,
            engine_->h_round_parameters,
            sizeof(rh_cuda::CandidateRoundParameters<T>),
            cudaMemcpyHostToDevice,
            engine_->stream
        ),
        "copy line-search round parameters"
    );
    const rh_cuda::CandidateRoundBuffers<T> buffers{
        typed<rh_cuda::CandidateRoundParameters<T>>(engine_->d_round_parameters),
        typed<T>(engine_->d_round_candidates),
        typed<T>(engine_->d_round_residuals),
        typed<double>(engine_->d_round_term_partials),
        typed<double>(engine_->d_round_loss_partials),
        typed<double>(engine_->d_round_results),
        typed<unsigned int>(engine_->d_round_counter),
    };
    check_cuda(
        rh_cuda::launch_candidate_round(
            buffers,
            typed<T>(engine_->d_trial_beta),
            typed<T>(engine_->d_direction),
            terms.gradient,
            typed<T>(engine_->d_coefficients),
            typed<T>(engine_->d_information),
            terms.penalty_sign,
            typed<T>(engine_->d_design),
            typed<T>(engine_->d_y),
            weights,
            rows,
            engine_->n_parameters,
            terms.penalized_count,
            tau,
            engine_->stream
        ),
        "launch line-search round"
    );
    check_cuda(
        cudaMemcpyAsync(
            engine_->h_round_results,
            engine_->d_round_results,
            rh_cuda::kCandidateRoundWidth * rh_cuda::kCandidateRoundSlots * sizeof(double),
            cudaMemcpyDeviceToHost,
            engine_->stream
        ),
        "read line-search round results"
    );
}

template <typename T>
bool CandidateRoundEvaluator::capture(
    int rows,
    const T* weights,
    T tau,
    const ObjectiveTerms<T>& terms
) {
    const cudaError_t begin = cudaStreamBeginCapture(
        engine_->stream, cudaStreamCaptureModeThreadLocal
    );
    if (begin != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    try {
        enqueue<T>(rows, weights, tau, terms);
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
void CandidateRoundEvaluator::evaluate(
    int rows,
    const T* weights,
    T tau,
    double n_total,
    const ObjectiveTerms<T>& terms,
    const CandidateRound& round,
    ObjectiveResult* results
) {
    if (round.width < 1 || round.width > rh_cuda::kCandidateRoundWidth) {
        fail(RH_CUDA_STATUS_INTERNAL_ERROR, "line-search round width is out of range");
    }
    // The previous round completed before its results were read, so the
    // pinned staging buffer is free to rewrite.
    auto* parameters = typed<rh_cuda::CandidateRoundParameters<T>>(engine_->h_round_parameters);
    parameters->width = round.width;
    parameters->form = round.form;
    for (int k = 0; k < rh_cuda::kCandidateRoundWidth; ++k) {
        parameters->scale[k] = static_cast<T>(round.scale[k]);
        parameters->threshold[k] = static_cast<T>(round.threshold[k]);
    }

    bool launched = false;
    if (graphs_) {
        if (execution_ == nullptr && !capture<T>(rows, weights, tau, terms)) {
            ++engine_->graph_fallbacks;
            engine_->enabled_flags &= ~RH_CUDA_ENGINE_FLAG_CUDA_GRAPHS;
            graphs_ = false;
        } else {
            check_cuda(cudaGraphLaunch(execution_, engine_->stream), "launch line-search round graph");
            if (launched_once_) {
                ++engine_->graph_replays;
            } else {
                launched_once_ = true;
            }
            launched = true;
        }
    }
    if (!launched) {
        enqueue<T>(rows, weights, tau, terms);
    }
    check_cuda(cudaStreamSynchronize(engine_->stream), "wait for line-search round");

    const double* slots = typed<double>(engine_->h_round_results);
    for (int k = 0; k < round.width; ++k) {
        const double* candidate_slots = slots + k * rh_cuda::kCandidateRoundSlots;
        ObjectiveResult& result = results[k];
        result.objective = (candidate_slots[0] + 0.5 * candidate_slots[1]) / n_total;
        if (terms.penalty_sign != nullptr) {
            result.objective -= terms.penalty_scale * candidate_slots[5];
        }
        result.difference_norm = std::sqrt(candidate_slots[2]);
        result.beta_norm = std::sqrt(candidate_slots[3]);
        result.gradient_dot = candidate_slots[4];
    }
}

template <typename T>
void compute_gradient_hessian(
    RhCudaEngine* engine,
    int rows,
    const T* beta,
    const T* residual,
    const T* weights,
    T tau,
    T bandwidth,
    double n_total,
    T ridge
) {
    const int parameters = static_cast<int>(engine->n_parameters);
    /*
     * solve_unpenalized evaluates the objective for the current trial before
     * every gradient/Hessian evaluation, and that round leaves the matching
     * residual in its residual buffer; recomputing y - X beta here would
     * duplicate a full-vector copy and GEMV on every Newton iteration.  The
     * kernel copies it into d_residual, so the next round may overwrite it.
     */
    check_cuda(
        rh_cuda::launch_residual_score_curvature(
            residual,
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
void compute_l1_gradient(
    RhCudaEngine* engine,
    int rows,
    const T* beta,
    const T* residual,
    const T* weights,
    T tau,
    double n_total,
    const T* penalty_sign,
    double penalty_scale
) {
    const int parameters = static_cast<int>(engine->n_parameters);
    // The proximal step needs no curvature, so only the weighted score is
    // formed from the residual the preceding round left.
    check_cuda(
        rh_cuda::launch_weighted_huber_score(
            residual,
            weights,
            typed<T>(engine->d_score),
            rows,
            tau,
            engine->stream
        ),
        "launch weighted Huber score"
    );
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
        "compute current L1 gradient"
    );
    check_cuda(
        rh_cuda::launch_subtract(
            beta, typed<T>(engine->d_coefficients), typed<T>(engine->d_delta), parameters, engine->stream
        ),
        "form L1 gradient coefficient delta"
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
        "add historical L1 gradient"
    );
    if (penalty_sign != nullptr) {
        check_cuda(
            rh_cuda::launch_axpby(
                typed<T>(engine->d_gradient),
                one,
                penalty_sign,
                static_cast<T>(-penalty_scale),
                typed<T>(engine->d_gradient),
                parameters,
                engine->stream
            ),
            "subtract L1 historical subgradient"
        );
    }
}

template <typename T>
void final_information(
    RhCudaEngine* engine,
    int rows,
    const T* residual,
    const T* weights,
    T tau,
    T bandwidth
) {
    const int parameters = static_cast<int>(engine->n_parameters);
    if (residual == nullptr) {
        compute_residual<T>(engine, rows, typed<T>(engine->d_trial_beta));
        residual = typed<T>(engine->d_residual);
    }
    check_cuda(
        rh_cuda::launch_residual_score_curvature(
            residual,
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
template void compute_l1_gradient<float>(
    RhCudaEngine*, int, const float*, const float*, const float*, float, double, const float*,
    double
);
template void compute_l1_gradient<double>(
    RhCudaEngine*, int, const double*, const double*, const double*, double, double,
    const double*, double
);
template void compute_gradient_hessian<float>(
    RhCudaEngine*, int, const float*, const float*, const float*, float, float, double, float
);
template void compute_gradient_hessian<double>(
    RhCudaEngine*, int, const double*, const double*, const double*, double, double, double,
    double
);
template void final_information<float>(
    RhCudaEngine*, int, const float*, const float*, float, float
);
template void final_information<double>(
    RhCudaEngine*, int, const double*, const double*, double, double
);
template void CandidateRoundEvaluator::evaluate<float>(
    int, const float*, float, double, const ObjectiveTerms<float>&, const CandidateRound&,
    ObjectiveResult*
);
template void CandidateRoundEvaluator::evaluate<double>(
    int, const double*, double, double, const ObjectiveTerms<double>&, const CandidateRound&,
    ObjectiveResult*
);
template const float* CandidateRoundEvaluator::candidate<float>(int) const;
template const double* CandidateRoundEvaluator::candidate<double>(int) const;
template const float* CandidateRoundEvaluator::residual<float>(int, int) const;
template const double* CandidateRoundEvaluator::residual<double>(int, int) const;

}  // namespace rh_cuda::engine
