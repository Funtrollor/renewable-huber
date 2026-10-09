#include "blas_traits.cuh"
#include "engine_internal.cuh"
#include "huber_kernels.cuh"
#include "workspace.cuh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

// RhCudaEngine is a global-scope type, so its members are defined at global
// scope too and cannot reach rh_cuda::engine by unqualified lookup. The
// using-directive keeps the moved body byte-identical; fine in a .cu, not in
// a header.
using namespace rh_cuda::engine;

RhCudaEngine::~RhCudaEngine() noexcept {
    if (device_id >= 0) {
        cudaSetDevice(device_id);
    }
    release(d_coefficients, stream, stream_ordered_allocations);
    release(d_information, stream, stream_ordered_allocations);
    release(d_information_next, stream, stream_ordered_allocations);
    release(d_trial_beta, stream, stream_ordered_allocations);
    release(d_delta, stream, stream_ordered_allocations);
    release(d_gradient, stream, stream_ordered_allocations);
    release(d_penalty_sign, stream, stream_ordered_allocations);
    release(d_direction, stream, stream_ordered_allocations);
    release(d_gram, stream, stream_ordered_allocations);
    release(d_hessian, stream, stream_ordered_allocations);
    release(d_factor, stream, stream_ordered_allocations);
    release(d_singular_values, stream, stream_ordered_allocations);
    release(d_svd_u, stream, stream_ordered_allocations);
    release(d_svd_v, stream, stream_ordered_allocations);
    release(d_svd_vector, stream, stream_ordered_allocations);
    release(d_factor_work, stream, stream_ordered_allocations);
    release(d_svd_work, stream, stream_ordered_allocations);
    release(d_reduction_results, stream, stream_ordered_allocations);
    release(d_round_parameters, stream, stream_ordered_allocations);
    release(d_round_candidates, stream, stream_ordered_allocations);
    release(d_round_term_partials, stream, stream_ordered_allocations);
    release(d_round_results, stream, stream_ordered_allocations);
    release(d_round_counter, stream, stream_ordered_allocations);
    void* pivots = d_pivots;
    release(pivots, stream, stream_ordered_allocations);
    d_pivots = nullptr;
    void* solver_info = d_solver_info;
    release(solver_info, stream, stream_ordered_allocations);
    d_solver_info = nullptr;
    release(d_design, stream, stream_ordered_allocations);
    release(d_y, stream, stream_ordered_allocations);
    release(d_weights, stream, stream_ordered_allocations);
    release(d_residual, stream, stream_ordered_allocations);
    release(d_score, stream, stream_ordered_allocations);
    release(d_curvature, stream, stream_ordered_allocations);
    release(d_weighted_design, stream, stream_ordered_allocations);
    release(d_round_residuals, stream, stream_ordered_allocations);
    release(d_round_loss_partials, stream, stream_ordered_allocations);
    if (stream_ordered_allocations && stream != nullptr) {
        cudaStreamSynchronize(stream);
    }
    if (h_solver_info != nullptr) {
        cudaFreeHost(h_solver_info);
        h_solver_info = nullptr;
    }
    if (h_reduction_results != nullptr) {
        cudaFreeHost(h_reduction_results);
        h_reduction_results = nullptr;
    }
    if (h_round_parameters != nullptr) {
        cudaFreeHost(h_round_parameters);
        h_round_parameters = nullptr;
    }
    if (h_round_results != nullptr) {
        cudaFreeHost(h_round_results);
        h_round_results = nullptr;
    }
    if (svd_params != nullptr) {
        cusolverDnDestroyGesvdjInfo(svd_params);
    }
    if (solver != nullptr) {
        cusolverDnDestroy(solver);
    }
    if (cublas != nullptr) {
        cublasDestroy(cublas);
    }
    if (cublas_reduction != nullptr) {
        cublasDestroy(cublas_reduction);
    }
    if (stream != nullptr) {
        cudaStreamDestroy(stream);
    }
}

namespace rh_cuda::engine {

template <typename T>
void allocate_static_buffers(RhCudaEngine* engine) {
    const size_t p = static_cast<size_t>(engine->n_parameters);
    const size_t square = checked_elements(engine->n_parameters, engine->n_parameters, "state matrix");

    const auto allocate_engine = [&](void** pointer, size_t elements, const char* name) {
        allocate<T>(
            pointer,
            elements,
            name,
            engine->stream,
            engine->stream_ordered_allocations,
            engine->memory_pool
        );
    };
    allocate_engine(&engine->d_coefficients, p, "coefficients");
    allocate_engine(&engine->d_information, square, "information");
    allocate_engine(&engine->d_information_next, square, "next information");
    allocate_engine(&engine->d_trial_beta, p, "trial coefficients");
    allocate_engine(&engine->d_delta, p, "coefficient delta");
    allocate_engine(&engine->d_gradient, p, "gradient");
    allocate_engine(&engine->d_penalty_sign, p, "L1 historical subgradient");
    allocate_engine(&engine->d_direction, p, "Newton direction");
    allocate_engine(&engine->d_gram, square, "weighted gram");
    allocate_engine(&engine->d_hessian, square, "Hessian");
    allocate_engine(&engine->d_factor, square, "factor matrix");
    allocate_engine(&engine->d_reduction_results, kReductionSlots, "device reduction results");
    const auto allocate_round = [&](auto element, void** pointer, size_t elements, const char* name) {
        allocate<decltype(element)>(
            pointer,
            elements,
            name,
            engine->stream,
            engine->stream_ordered_allocations,
            engine->memory_pool
        );
    };
    const size_t round_width = static_cast<size_t>(rh_cuda::kCandidateRoundWidth);
    allocate_engine(&engine->d_round_candidates, p * round_width, "line-search round candidates");
    allocate_round(
        double{},
        &engine->d_round_term_partials,
        round_width * static_cast<size_t>(rh_cuda::candidate_round_chunks(engine->n_parameters)) *
            rh_cuda::kCandidateRoundTermSlots,
        "line-search round term partials"
    );
    allocate_round(
        double{},
        &engine->d_round_results,
        round_width * rh_cuda::kCandidateRoundSlots,
        "line-search round results"
    );
    allocate_round(
        static_cast<unsigned char>(0),
        &engine->d_round_parameters,
        sizeof(rh_cuda::CandidateRoundParameters<T>),
        "line-search round parameters"
    );
    allocate_round(0u, &engine->d_round_counter, 1, "line-search round counter");
    allocate<int>(
        reinterpret_cast<void**>(&engine->d_pivots),
        p,
        "LU pivots",
        engine->stream,
        engine->stream_ordered_allocations,
        engine->memory_pool
    );
    allocate<int>(
        reinterpret_cast<void**>(&engine->d_solver_info),
        2,
        "solver info",
        engine->stream,
        engine->stream_ordered_allocations,
        engine->memory_pool
    );
    check_cuda(
        cudaMallocHost(reinterpret_cast<void**>(&engine->h_solver_info), 2 * sizeof(int)),
        "pinned host solver info"
    );
    check_cuda(
        cudaMallocHost(&engine->h_reduction_results, kReductionSlots * sizeof(double)),
        "pinned host objective reductions"
    );
    check_cuda(
        cudaMallocHost(&engine->h_round_parameters, sizeof(rh_cuda::CandidateRoundParameters<T>)),
        "pinned host line-search round parameters"
    );
    check_cuda(
        cudaMallocHost(
            &engine->h_round_results,
            round_width * rh_cuda::kCandidateRoundSlots * sizeof(double)
        ),
        "pinned host line-search round results"
    );

    const int n = static_cast<int>(engine->n_parameters);
    int cholesky_lwork = 0;
    int lu_lwork = 0;
    check_cusolver(
        Solver<T>::potrf_buffer_size(
            engine->solver, n, typed<T>(engine->d_factor), n, &cholesky_lwork
        ),
        "cusolverDn*potrf_bufferSize"
    );
    check_cusolver(
        Solver<T>::getrf_buffer_size(
            engine->solver, n, typed<T>(engine->d_factor), n, &lu_lwork
        ),
        "cusolverDn*getrf_bufferSize"
    );
    engine->factor_lwork = std::max(cholesky_lwork, lu_lwork);
    allocate<T>(
        &engine->d_factor_work,
        static_cast<size_t>(engine->factor_lwork),
        "dense factorization workspace",
        engine->stream,
        engine->stream_ordered_allocations,
        engine->memory_pool
    );
    check_cuda(cudaMemsetAsync(engine->d_coefficients, 0, p * sizeof(T), engine->stream), "zero coefficients");
    check_cuda(cudaMemsetAsync(engine->d_information, 0, square * sizeof(T), engine->stream), "zero information");
    // The residual kernel's last block counts arrivals from zero and resets
    // the counter itself, so it is zeroed exactly once.
    check_cuda(
        cudaMemsetAsync(engine->d_round_counter, 0, sizeof(unsigned int), engine->stream),
        "zero line-search round counter"
    );
    check_cuda(cudaStreamSynchronize(engine->stream), "initial state synchronization");
}
void release_batch_buffers(RhCudaEngine* engine) noexcept {
    release(engine->d_design, engine->stream, engine->stream_ordered_allocations);
    release(engine->d_y, engine->stream, engine->stream_ordered_allocations);
    release(engine->d_weights, engine->stream, engine->stream_ordered_allocations);
    release(engine->d_residual, engine->stream, engine->stream_ordered_allocations);
    release(engine->d_score, engine->stream, engine->stream_ordered_allocations);
    release(engine->d_curvature, engine->stream, engine->stream_ordered_allocations);
    release(engine->d_weighted_design, engine->stream, engine->stream_ordered_allocations);
    release(engine->d_round_residuals, engine->stream, engine->stream_ordered_allocations);
    release(engine->d_round_loss_partials, engine->stream, engine->stream_ordered_allocations);
    engine->capacity_rows = 0;
}
template <typename T>
void ensure_batch_capacity(RhCudaEngine* engine, int64_t rows) {
    if (rows <= engine->capacity_rows) {
        return;
    }
    release_batch_buffers(engine);
    const size_t vector = static_cast<size_t>(rows);
    const size_t matrix = checked_elements(rows, engine->n_parameters, "batch design");
    const auto allocate_batch = [&](void** pointer, size_t elements, const char* name) {
        allocate<T>(
            pointer,
            elements,
            name,
            engine->stream,
            engine->stream_ordered_allocations,
            engine->memory_pool
        );
    };
    allocate_batch(&engine->d_design, matrix, "batch design");
    allocate_batch(&engine->d_y, vector, "batch target");
    allocate_batch(&engine->d_weights, vector, "batch weights");
    allocate_batch(&engine->d_residual, vector, "residual");
    allocate_batch(&engine->d_score, vector, "score");
    allocate_batch(&engine->d_curvature, vector, "curvature");
    allocate_batch(&engine->d_weighted_design, matrix, "weighted design");
    const size_t round_width = static_cast<size_t>(rh_cuda::kCandidateRoundWidth);
    allocate_batch(
        &engine->d_round_residuals,
        checked_elements(rows, rh_cuda::kCandidateRoundWidth, "line-search round residuals"),
        "line-search round residuals"
    );
    allocate<double>(
        &engine->d_round_loss_partials,
        round_width * static_cast<size_t>(rh_cuda::candidate_round_blocks(rows)),
        "line-search round loss partials",
        engine->stream,
        engine->stream_ordered_allocations,
        engine->memory_pool
    );
    engine->capacity_rows = rows;
}
template <typename T>
void ensure_svd_workspace(RhCudaEngine* engine) {
    if (engine->svd_params != nullptr) {
        return;
    }

    /*
     * The normal renewable Hessian is SPD and uses POTRF/POTRS. Creating
     * gesvdj metadata and allocating U/V/workspace during every cold engine
     * construction cost several milliseconds even when the fallback was
     * never reached. Keep the exact minimum-norm path, but initialize its
     * resources only after LU has actually reported singularity.
     */
    const size_t p = static_cast<size_t>(engine->n_parameters);
    const size_t square = checked_elements(
        engine->n_parameters, engine->n_parameters, "lazy SVD matrix"
    );
    const auto allocate_svd = [&](void** pointer, size_t elements, const char* name) {
        allocate<T>(
            pointer,
            elements,
            name,
            engine->stream,
            engine->stream_ordered_allocations,
            engine->memory_pool
        );
    };
    allocate_svd(&engine->d_singular_values, p, "singular values");
    allocate_svd(&engine->d_svd_u, square, "SVD U");
    allocate_svd(&engine->d_svd_v, square, "SVD V");
    allocate_svd(&engine->d_svd_vector, p, "SVD vector");

    check_cusolver(
        cusolverDnCreateGesvdjInfo(&engine->svd_params),
        "cusolverDnCreateGesvdjInfo"
    );
    check_cusolver(
        cusolverDnXgesvdjSetTolerance(
            engine->svd_params,
            static_cast<double>(std::numeric_limits<T>::epsilon())
        ),
        "cusolverDnXgesvdjSetTolerance"
    );
    check_cusolver(
        cusolverDnXgesvdjSetMaxSweeps(engine->svd_params, 100),
        "cusolverDnXgesvdjSetMaxSweeps"
    );

    const int n = static_cast<int>(engine->n_parameters);
    check_cusolver(
        Solver<T>::gesvdj_buffer_size(
            engine->solver,
            typed<T>(engine->d_factor),
            typed<T>(engine->d_singular_values),
            typed<T>(engine->d_svd_u),
            typed<T>(engine->d_svd_v),
            n,
            &engine->svd_lwork,
            engine->svd_params
        ),
        "cusolverDn*gesvdj_bufferSize"
    );
    allocate_svd(
        &engine->d_svd_work,
        static_cast<size_t>(engine->svd_lwork),
        "SVD workspace"
    );
}

// Explicit instantiation: the engine is only ever float or double, and a
// missing pair fails the link instead of silently duplicating a definition.
template void allocate_static_buffers<float>(RhCudaEngine*);
template void allocate_static_buffers<double>(RhCudaEngine*);
template void ensure_batch_capacity<float>(RhCudaEngine*, int64_t);
template void ensure_batch_capacity<double>(RhCudaEngine*, int64_t);
template void ensure_svd_workspace<float>(RhCudaEngine*);
template void ensure_svd_workspace<double>(RhCudaEngine*);

}  // namespace rh_cuda::engine
