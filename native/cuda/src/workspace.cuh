#ifndef RENEWABLE_HUBER_RH_CUDA_WORKSPACE_CUH
#define RENEWABLE_HUBER_RH_CUDA_WORKSPACE_CUH

/*
 * Every device allocation the engine owns, and the destructor that releases
 * them.
 *
 * Putting the release list next to the allocation it mirrors is the point of
 * this file: in engine.cu the two sat 200 lines apart, so "every allocated
 * pointer is freed exactly once" could not be checked by reading.
 */

#include "engine_state.cuh"

#include <cstdint>

namespace rh_cuda::engine {

/// Allocate the p-sized and p*p-sized buffers that persist across batches.
template <typename T>
void allocate_static_buffers(RhCudaEngine* engine);

/// Release the per-batch buffers and reset the row capacity.
void release_batch_buffers(RhCudaEngine* engine) noexcept;

/// Grow the per-batch buffers if this batch has more rows than the last.
template <typename T>
void ensure_batch_capacity(RhCudaEngine* engine, int64_t rows);

/// Initialize the SVD handle and buffers on first use.  The SVD fallback is
/// reached only by rank-deficient or ill-conditioned systems, so its
/// comparatively expensive workspace stays unallocated until then.
template <typename T>
void ensure_svd_workspace(RhCudaEngine* engine);

}  // namespace rh_cuda::engine

#endif  // RENEWABLE_HUBER_RH_CUDA_WORKSPACE_CUH
