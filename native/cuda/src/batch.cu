#include "blas_traits.cuh"
#include "engine_internal.cuh"
#include "batch.cuh"
#include "huber_kernels.cuh"
#include "workspace.cuh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace rh_cuda::engine {

void validate_intercept_layout(int64_t n_features_in, const RhCudaEngine* engine) {
    // n_parameters is either n_features_in (no intercept) or n_features_in + 1
    // (intercept).  Any wider gap would let the device-side intercept append
    // fill only part of d_design and leave the rest stale, so reject it here
    // rather than solving against uninitialized columns.  The difference is
    // taken rather than n_features_in + 1 so an extreme n_features_in cannot
    // overflow before it is rejected.
    const int64_t intercept_gap = engine->n_parameters - n_features_in;
    if (n_features_in <= 0 || (intercept_gap != 0 && intercept_gap != 1)) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "n_features_in is incompatible with engine state");
    }
}

void validate_config(const RhCudaUpdateConfig* config, const RhCudaEngine* engine) {
    check_header(config, "update config");
    validate_intercept_layout(config->n_features_in, engine);
    if (config->max_iter < 1 || config->max_iter > std::numeric_limits<int>::max()) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "max_iter must be in the CUDA solver range");
    }
    if (!finite_positive(config->tau) || !finite_positive(config->bandwidth_scale) ||
        !finite_positive(config->tolerance) || !finite_nonnegative(config->ridge)) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "update config contains an invalid numerical value");
    }
    // Fail closed: an unknown penalty must never be solved as if it were none.
    if (config->penalty != RH_CUDA_PENALTY_NONE && config->penalty != RH_CUDA_PENALTY_L1) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "update config names an unknown penalty");
    }
    if (config->reserved0 != 0) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "update config reserved0 must be zero");
    }
    if (!finite_nonnegative(config->lambda_scale)) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "lambda_scale must be finite and non-negative");
    }
}


void validate_batch(
    const BatchView& batch,
    const RhCudaUpdateConfig* config,
    const RhCudaEngine* engine
) {
    if (batch.x_design == nullptr || batch.y == nullptr) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "x_design and y must not be null");
    }
    if (batch.n_rows <= 0 || batch.n_rows > std::numeric_limits<int>::max()) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "n_rows must be in the CUDA BLAS range");
    }
    if (batch.n_columns != engine->n_parameters && batch.n_columns != config->n_features_in) {
        fail(
            RH_CUDA_STATUS_INVALID_ARGUMENT,
            "feature columns must match n_features_in or the expanded engine parameters"
        );
    }
    if (!finite_positive(batch.batch_weight)) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "batch_weight must be finite and positive");
    }
}

double bandwidth_for(const RhCudaEngine* engine, double batch_weight, const RhCudaUpdateConfig* config) {
    const double n_total = engine->weight_sum + batch_weight;
    if (!finite_positive(n_total)) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "cumulative sample weight must be finite and positive");
    }
    const double predictors = static_cast<double>(std::max<int64_t>(config->n_features_in, 2));
    const double raw = config->bandwidth_scale / (std::sqrt(n_total) * std::log(predictors));
    return std::min(raw, config->tau);
}

double lambda_for(const RhCudaEngine* engine, double batch_weight, const RhCudaUpdateConfig* config) {
    if (config->penalty != RH_CUDA_PENALTY_L1) {
        return 0.0;
    }
    const double n_total = engine->weight_sum + batch_weight;
    if (!finite_positive(n_total)) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "cumulative sample weight must be finite and positive");
    }
    const double predictors = static_cast<double>(std::max<int64_t>(config->n_features_in, 2));
    return config->lambda_scale * config->tau * std::sqrt(std::log(predictors) / n_total);
}

void validate_device_pointer(const RhCudaEngine* engine, const void* pointer, const char* name) {
    if (pointer == nullptr) {
        return;
    }
    cudaPointerAttributes attributes{};
    check_cuda(cudaPointerGetAttributes(&attributes, pointer), name);
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != engine->device_id) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "device batch pointer is not on the engine CUDA device");
    }
}

template <typename T>
void copy_batch(RhCudaEngine* engine, const BatchView& batch) {
    ensure_batch_capacity<T>(engine, batch.n_rows);
    const size_t matrix = checked_elements(batch.n_rows, batch.n_columns, "batch features");
    const size_t vector = static_cast<size_t>(batch.n_rows);
    if (batch.copy_kind == cudaMemcpyDeviceToDevice) {
        validate_device_pointer(engine, batch.x_design, "inspect device X");
        validate_device_pointer(engine, batch.y, "inspect device y");
        validate_device_pointer(engine, batch.sample_weight, "inspect device sample_weight");
    }
    if (batch.n_columns == engine->n_parameters) {
        check_cuda(
            cudaMemcpyAsync(
                engine->d_design,
                batch.x_design,
                matrix * sizeof(T),
                batch.copy_kind,
                engine->stream
            ),
            "copy X_design into engine workspace"
        );
    } else {
        const T* unexpanded = typed<T>(batch.x_design);
        if (batch.copy_kind != cudaMemcpyDeviceToDevice) {
            check_cuda(
                cudaMemcpyAsync(
                    engine->d_weighted_design,
                    batch.x_design,
                    matrix * sizeof(T),
                    batch.copy_kind,
                    engine->stream
                ),
                "copy unexpanded features into engine workspace"
            );
            unexpanded = typed<T>(engine->d_weighted_design);
        }
        check_cuda(
            rh_cuda::launch_append_intercept(
                unexpanded,
                typed<T>(engine->d_design),
                batch.n_rows,
                batch.n_columns,
                engine->stream
            ),
            "append intercept column on device"
        );
    }
    if (batch.copy_kind != cudaMemcpyDeviceToDevice) {
        check_cuda(
            cudaMemcpyAsync(engine->d_y, batch.y, vector * sizeof(T), batch.copy_kind, engine->stream),
            "copy y into engine workspace"
        );
    }
    if (batch.sample_weight != nullptr) {
        check_cuda(
            cudaMemcpyAsync(
                engine->d_weights, batch.sample_weight, vector * sizeof(T), batch.copy_kind, engine->stream
            ),
            "copy sample_weight into engine workspace"
        );
    }
}

// Explicit instantiation: the engine is only ever float or double, and a
// missing pair fails the link instead of silently duplicating a definition.
template void copy_batch<float>(RhCudaEngine*, const BatchView&);
template void copy_batch<double>(RhCudaEngine*, const BatchView&);

}  // namespace rh_cuda::engine
