#ifndef RENEWABLE_HUBER_RH_CUDA_ENGINE_STATE_CUH
#define RENEWABLE_HUBER_RH_CUDA_ENGINE_STATE_CUH

/*
 * The resident engine type and the few zero-cost helpers that name it.
 *
 * RhCudaEngine stays at global scope. rh_cuda.h forward-declares
 * `struct RhCudaEngine` there and all 17 extern "C" entry points take
 * `::RhCudaEngine*`; defining it inside a namespace would create a second,
 * unrelated type and leave the ABI's one permanently incomplete.
 *
 * Keep this header to the state type alone. Anything that *operates on* an
 * engine belongs in a module header -- otherwise every TU recompiles whenever
 * any helper changes, which is engine.cu rebuilt as a header.
 */

#include "engine_internal.cuh"
#include "rh_cuda.h"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <cstdint>

struct RhCudaEngine {
    RhCudaDType dtype = RH_CUDA_DTYPE_FLOAT64;
    int32_t device_id = 0;
    int64_t n_parameters = 0;
    int64_t capacity_rows = 0;

    int64_t n_samples_seen = 0;
    int64_t batch_count = 0;
    double previous_lambda = 0.0;
    double weight_sum = 0.0;
    bool information_is_symmetric = true;
    bool stream_ordered_allocations = false;
    cudaMemPool_t memory_pool = nullptr;
    uint64_t requested_flags = 0;
    uint64_t enabled_flags = 0;
    uint64_t graph_captures = 0;
    uint64_t graph_replays = 0;
    uint64_t graph_fallbacks = 0;

    cudaStream_t stream = nullptr;
    cublasHandle_t cublas = nullptr;
    cublasHandle_t cublas_reduction = nullptr;
    cusolverDnHandle_t solver = nullptr;
    gesvdjInfo_t svd_params = nullptr;

    void* d_coefficients = nullptr;
    void* d_information = nullptr;
    void* d_information_next = nullptr;
    void* d_trial_beta = nullptr;
    void* d_candidate = nullptr;
    void* d_delta = nullptr;
    void* d_history_vector = nullptr;
    void* d_gradient = nullptr;
    void* d_direction = nullptr;
    void* d_gram = nullptr;
    void* d_hessian = nullptr;
    void* d_factor = nullptr;
    void* d_singular_values = nullptr;
    void* d_svd_u = nullptr;
    void* d_svd_v = nullptr;
    void* d_svd_vector = nullptr;
    void* d_factor_work = nullptr;
    void* d_svd_work = nullptr;
    void* d_reduction_results = nullptr;
    int* d_pivots = nullptr;
    int* d_solver_info = nullptr;
    int* h_solver_info = nullptr;
    void* h_reduction_results = nullptr;
    int factor_lwork = 0;
    int svd_lwork = 0;

    void* d_design = nullptr;
    void* d_y = nullptr;
    void* d_weights = nullptr;
    void* d_residual = nullptr;
    void* d_score = nullptr;
    void* d_curvature = nullptr;
    void* d_loss = nullptr;
    void* d_weighted_design = nullptr;

    rh_cuda::engine::ErrorBuffer last_error{};

    // Defined out of line in workspace.cu so its 28-entry release list sits
    // next to the allocation it mirrors.
    ~RhCudaEngine() noexcept;
};

namespace rh_cuda::engine {

template <typename T>
T* typed(void* value) {
    return static_cast<T*>(value);
}

template <typename T>
const T* typed(const void* value) {
    return static_cast<const T*>(value);
}

struct BatchView {
    const void* x_design;
    const void* y;
    const void* sample_weight;
    int64_t n_rows;
    int64_t n_columns;
    double batch_weight;
    cudaMemcpyKind copy_kind;
};

class ScopedPointerAlias {
public:
    ScopedPointerAlias(void** slot, const void* replacement, bool enabled)
        : slot_(enabled ? slot : nullptr), original_(enabled ? *slot : nullptr) {
        if (slot_ != nullptr) {
            *slot_ = const_cast<void*>(replacement);
        }
    }

    ~ScopedPointerAlias() {
        if (slot_ != nullptr) {
            *slot_ = original_;
        }
    }

    ScopedPointerAlias(const ScopedPointerAlias&) = delete;
    ScopedPointerAlias& operator=(const ScopedPointerAlias&) = delete;

private:
    void** slot_;
    void* original_;
};

}  // namespace rh_cuda::engine

#endif  // RENEWABLE_HUBER_RH_CUDA_ENGINE_STATE_CUH
