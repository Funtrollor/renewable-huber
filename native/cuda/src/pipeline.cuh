#ifndef RENEWABLE_HUBER_RH_CUDA_PIPELINE_CUH
#define RENEWABLE_HUBER_RH_CUDA_PIPELINE_CUH

/*
 * One complete batch transition, and prediction.
 *
 * Prediction lives here rather than in a file of its own: it is 50 lines, and
 * it shares ensure_batch_capacity, d_design and d_residual with the update
 * path.  Separating the two would hide d_residual's dual role rather than
 * clarify it.
 */

#include "engine_state.cuh"

#include <cstdint>

namespace rh_cuda::engine {

/// Enqueue the device-to-host snapshot of committed state, converting the
/// engine's column-major layout to the portable row-major checkpoint form.
template <typename T>
void enqueue_state_copy(
    RhCudaEngine* engine,
    const T* coefficients,
    const T* information,
    RhCudaHostState* state
);

/// Copy the scalar renewable counters into a caller-owned host state.
void fill_state_metadata(const RhCudaEngine* engine, RhCudaHostState* state);

/// Run one complete batch transition: damped Newton for penalty NONE, LAMM
/// proximal gradient for penalty L1.
///
/// State is committed transactionally: the solve writes into staging buffers,
/// and the active pointers are swapped only after the single stream
/// synchronization reports success, so a failure leaves the previous state
/// intact rather than half applied.  Passing `exported_state` folds the
/// portable checkpoint copy into that same synchronization.
template <typename T>
RhCudaStatus update_typed(
    RhCudaEngine* engine,
    const BatchView& batch,
    const RhCudaUpdateConfig* config,
    RhCudaDiagnostics* diagnostics,
    RhCudaHostState* exported_state = nullptr
);

/// Predict into a caller-owned host buffer from the resident coefficients.
/// Host input is staged into d_design; device input is read in place when it
/// is already expanded, or widened on device when it is not.
template <typename T>
RhCudaStatus predict_typed(RhCudaEngine* engine, const RhCudaPrediction* request);

}  // namespace rh_cuda::engine

#endif  // RENEWABLE_HUBER_RH_CUDA_PIPELINE_CUH
