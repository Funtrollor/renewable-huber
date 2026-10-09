#ifndef RENEWABLE_HUBER_HUBER_KERNELS_CUH
#define RENEWABLE_HUBER_HUBER_KERNELS_CUH

#include <cuda_runtime_api.h>

#include <stdint.h>

namespace rh_cuda {

template <typename T>
cudaError_t launch_append_intercept(
    const T* features,
    T* design,
    int64_t rows,
    int64_t feature_columns,
    cudaStream_t stream
);

template <typename T>
cudaError_t launch_residual_score_curvature(
    const T* residual_input,
    T* residual,
    T* score,
    T* curvature,
    int64_t count,
    T tau,
    T bandwidth,
    cudaStream_t stream
);

template <typename T>
cudaError_t launch_weight_score(
    T* score,
    const T* weights,
    int64_t count,
    cudaStream_t stream
);

template <typename T>
cudaError_t launch_weight_design(
    const T* design,
    const T* curvature,
    const T* weights,
    T* weighted_design,
    int64_t rows,
    int64_t columns,
    cudaStream_t stream
);

template <typename T>
cudaError_t launch_subtract(
    const T* left,
    const T* right,
    T* output,
    int64_t count,
    cudaStream_t stream
);

template <typename T>
cudaError_t launch_add_scaled_identity(
    const T* matrix,
    T* output,
    int64_t side,
    T scale,
    cudaStream_t stream
);

template <typename T>
cudaError_t launch_add_matrix(
    const T* left,
    const T* right,
    T* output,
    int64_t count,
    cudaStream_t stream
);

template <typename T>
cudaError_t launch_scale_and_add_identity(
    T* matrix,
    int64_t side,
    T scale,
    T ridge,
    cudaStream_t stream
);

template <typename T>
cudaError_t launch_axpby(
    const T* left,
    T left_scale,
    const T* right,
    T right_scale,
    T* output,
    int64_t count,
    cudaStream_t stream
);

template <typename T>
cudaError_t launch_copy(T* destination, const T* source, int64_t count, cudaStream_t stream);

template <typename T>
cudaError_t launch_pseudoinverse_scale(
    T* vector,
    const T* singular_values,
    int64_t count,
    T cutoff,
    cudaStream_t stream
);

template <typename T>
cudaError_t launch_transpose(
    const T* input,
    T* output,
    int64_t rows,
    int64_t columns,
    cudaStream_t stream
);

template <typename T>
cudaError_t launch_mirror_lower_triangle(T* matrix, int64_t side, cudaStream_t stream);

/*
 * L1 building blocks.  `penalized_count` is the number of leading coordinates
 * the penalty applies to: n_parameters, or n_parameters - 1 when the trailing
 * coordinate is the unpenalized intercept.
 */

/// output[i] = sign(coefficients[i]) for penalized coordinates, else 0.
template <typename T>
cudaError_t launch_penalty_sign(
    const T* coefficients,
    T* output,
    int64_t count,
    int64_t penalized_count,
    cudaStream_t stream
);

/// score[i] = clamp(residual[i], -tau, tau) * weights[i] (weights may be null).
template <typename T>
cudaError_t launch_weighted_huber_score(
    const T* residual,
    const T* weights,
    T* score,
    int64_t count,
    T tau,
    cudaStream_t stream
);

/*
 * Fused line-search rounds.
 *
 * A round evaluates up to kCandidateRoundWidth candidates along one search
 * direction -- Newton steps 2^-b, 2^-(b+1), ... or LAMM curvatures phi,
 * 2 phi, ... -- in two kernels, so the host pays one transfer and one stream
 * synchronization per round instead of about twenty API calls and a
 * synchronization per candidate.
 *
 * Every per-candidate value is computed by the same threads in the same order
 * whatever the round width, and every reduction runs in a fixed order over a
 * grid that depends only on the batch shape. Which candidates share a round
 * therefore never changes a result bit, and repeated runs are bit-identical.
 * Sums are accumulated in double for both dtypes.
 */
constexpr int kCandidateRoundWidth = 4;

/// double slots per candidate in a round's results:
///   0 weighted Huber loss          1 (c - history)' J (c - history)
///   2 ||c - beta||^2               3 ||c||^2
///   4 gradient . (c - beta)        5 sign(history) . (c - history)
constexpr int kCandidateRoundSlots = 6;

/// Slots 1..5 are reduced over coefficients rather than rows.
constexpr int kCandidateRoundTermSlots = kCandidateRoundSlots - 1;

/// How a round forms candidate k from beta.
enum CandidateForm : int32_t {
    /// c = beta; evaluates the objective at the current point.
    kCandidateCopy = 0,
    /// c = beta - scale[k] * direction.
    kCandidateNewton = 1,
    /// c = soft_threshold(beta - gradient * scale[k], threshold[k]) on the
    /// penalized coordinates; scale[k] = 1 / phi_k.
    kCandidateProximal = 2,
};

/// Per-round inputs, read by the kernels from device memory so one captured
/// CUDA Graph serves every round of an update.
template <typename T>
struct CandidateRoundParameters {
    int32_t width;
    int32_t form;
    T scale[kCandidateRoundWidth];
    T threshold[kCandidateRoundWidth];
};

/// Residual-kernel blocks for a batch of `rows`. A function of the shape
/// alone, so the reduction order never depends on the device or the round.
int candidate_round_blocks(int64_t rows);

/// Term-kernel blocks per candidate for `parameters` coefficients.
int candidate_round_chunks(int64_t parameters);

template <typename T>
struct CandidateRoundBuffers {
    const CandidateRoundParameters<T>* parameters;
    /// n_parameters x kCandidateRoundWidth, one column per candidate.
    T* candidates;
    /// rows x kCandidateRoundWidth, one column per candidate.
    T* residuals;
    /// kCandidateRoundWidth x candidate_round_chunks(p) x kCandidateRoundTermSlots.
    double* term_partials;
    /// kCandidateRoundWidth x candidate_round_blocks(rows).
    double* loss_partials;
    /// kCandidateRoundWidth x kCandidateRoundSlots.
    double* results;
    /// Zero between rounds; the last residual block resets it.
    unsigned int* counter;
};

/// Enqueue one round: form the candidates, reduce the coefficient-space
/// terms, then the residuals and Huber loss, and write `results`.
/// `gradient` may be null unless the form is proximal; `penalty_sign` and
/// `weights` may be null.
template <typename T>
cudaError_t launch_candidate_round(
    const CandidateRoundBuffers<T>& buffers,
    const T* beta,
    const T* direction,
    const T* gradient,
    const T* coefficients,
    const T* information,
    const T* penalty_sign,
    const T* design,
    const T* y,
    const T* weights,
    int64_t rows,
    int64_t parameters,
    int64_t penalized_count,
    T tau,
    cudaStream_t stream
);

}  // namespace rh_cuda

#endif  // RENEWABLE_HUBER_HUBER_KERNELS_CUH
