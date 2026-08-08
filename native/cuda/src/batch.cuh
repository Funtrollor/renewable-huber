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

/// Reject a configuration that cannot describe this engine's parameter shape.
void validate_config(const RhCudaUnpenalizedConfig* config, const RhCudaEngine* engine);

/// Reject a batch whose pointers, row count, or column width are unusable.
void validate_batch(
    const BatchView& batch,
    const RhCudaUnpenalizedConfig* config,
    const RhCudaEngine* engine
);

/// Smoothing bandwidth for this batch, given the accumulated weight.
double bandwidth_for(
    const RhCudaEngine* engine,
    double batch_weight,
    const RhCudaUnpenalizedConfig* config
);

/// Stage the batch into the engine's device buffers.  A batch submitted at
/// n_features_in width has its trailing all-ones intercept column appended on
/// device; one submitted at n_parameters width is copied as is.
template <typename T>
void copy_batch(RhCudaEngine* engine, const BatchView& batch);

}  // namespace rh_cuda::engine

#endif  // RENEWABLE_HUBER_RH_CUDA_BATCH_CUH
