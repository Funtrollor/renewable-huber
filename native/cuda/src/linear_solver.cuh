#ifndef RENEWABLE_HUBER_RH_CUDA_LINEAR_SOLVER_CUH
#define RENEWABLE_HUBER_RH_CUDA_LINEAR_SOLVER_CUH

/*
 * The Cholesky -> LU -> lazy SVD ladder that produces each Newton direction.
 *
 * Only solve_direction and the deferred Cholesky check are exported; the LU
 * and minimum-norm SVD fallbacks are reached solely through them and stay
 * internal to this translation unit.
 */

#include "engine_state.cuh"

#include <cstdint>

namespace rh_cuda::engine {

/// Solve for the next Newton direction.  A well-conditioned positive-definite
/// system takes the Cholesky path; failure falls back to pivoted LU and then,
/// only for rank-deficient or ill-conditioned systems, to a minimum-norm SVD
/// whose comparatively expensive workspace is allocated on first use.
///
/// Returns true when a Cholesky factorization was enqueued and its `info`
/// result has not been inspected yet; the caller confirms it later through
/// cholesky_candidate_is_valid, which is what keeps the fast path free of a
/// mid-iteration synchronization.
template <typename T>
bool solve_direction(RhCudaEngine* engine, bool allow_cholesky, bool* used_fallback);

/// Inspect the deferred Cholesky `info` and report whether the direction it
/// produced is usable.
template <typename T>
bool cholesky_candidate_is_valid(RhCudaEngine* engine, bool* used_fallback);

}  // namespace rh_cuda::engine

#endif  // RENEWABLE_HUBER_RH_CUDA_LINEAR_SOLVER_CUH
