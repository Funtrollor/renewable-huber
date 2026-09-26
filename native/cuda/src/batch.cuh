#ifndef RENEWABLE_HUBER_RH_CUDA_BATCH_CUH
#define RENEWABLE_HUBER_RH_CUDA_BATCH_CUH

/*
 * Validation and staging for one submitted batch.
 *
 * Split out on purpose rather than folded into the update pipeline: this is
 * where the accepted-column-width contract lives, including the device-side
 * intercept append that the C header used to describe incorrectly. Giving it
 * its own translation unit keeps that contract visible and gives the smoke
 * test an obvious target.
 */

#include "engine_state.cuh"

#include <cstdint>

namespace rh_cuda::engine {

/// Reject an intercept layout other than n_parameters == n_features_in (+1).
void validate_intercept_layout(int64_t n_features_in, const RhCudaEngine* engine);

/// Reject a configuration that cannot describe this engine's parameter shape,
/// names an unknown penalty, or carries a non-zero reserved field.
void validate_config(const RhCudaUpdateConfig* config, const RhCudaEngine* engine);

/// Reject a batch whose pointers, row count, or column width are unusable.
void validate_batch(
    const BatchView& batch,
    const RhCudaUpdateConfig* config,
    const RhCudaEngine* engine
);

/// Smoothing bandwidth for this batch, given the accumulated weight.
double bandwidth_for(
    const RhCudaEngine* engine,
    double batch_weight,
    const RhCudaUpdateConfig* config
);

/// L1 regularization level for this batch; zero for an unpenalized update.
double lambda_for(
    const RhCudaEngine* engine,
    double batch_weight,
    const RhCudaUpdateConfig* config
);

/// Require that a non-null pointer addresses device memory on the engine's
/// CUDA device.  Used for every caller-owned device input.
void validate_device_pointer(const RhCudaEngine* engine, const void* pointer, const char* name);

/// Stage the batch into the engine's device buffers.  A batch submitted at
/// n_features_in width has its trailing all-ones intercept column appended on
/// device; one submitted at n_parameters width is copied as is.
template <typename T>
void copy_batch(RhCudaEngine* engine, const BatchView& batch);

}  // namespace rh_cuda::engine

#endif  // RENEWABLE_HUBER_RH_CUDA_BATCH_CUH
