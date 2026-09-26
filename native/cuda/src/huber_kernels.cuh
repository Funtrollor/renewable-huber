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
cudaError_t launch_huber_loss(
    const T* residual,
    const T* weights,
    T* loss,
    int64_t count,
    T tau,
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
cudaError_t launch_candidate(
    const T* beta,
    const T* direction,
    T step,
    T* candidate,
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

/// candidate[i] = soft_threshold(beta[i] - gradient[i] * inverse_phi, t_i),
/// with t_i = threshold for penalized coordinates and 0 otherwise.
template <typename T>
cudaError_t launch_soft_threshold_candidate(
    const T* beta,
    const T* gradient,
    T inverse_phi,
    T threshold,
    T* candidate,
    int64_t count,
    int64_t penalized_count,
    cudaStream_t stream
);

}  // namespace rh_cuda

#endif  // RENEWABLE_HUBER_HUBER_KERNELS_CUH
