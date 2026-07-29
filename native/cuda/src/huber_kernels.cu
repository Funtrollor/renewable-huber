#include "huber_kernels.cuh"

#include <algorithm>

namespace rh_cuda {
namespace {

constexpr int kThreadsPerBlock = 256;

inline int blocks_for(int64_t count) {
    return static_cast<int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
}

template <typename T>
__global__ void residual_score_curvature_kernel(
    const T* residual_input,
    T* residual,
    T* score,
    T* curvature,
    int64_t count,
    T tau,
    T bandwidth
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    const T value = residual_input[index];
    const T h = bandwidth < tau ? bandwidth : tau;
    residual[index] = value;
    score[index] = value < -tau ? -tau : (value > tau ? tau : value);

    if (value < -tau - h || value > tau + h) {
        curvature[index] = static_cast<T>(0);
    } else if (value <= -tau + h) {
        curvature[index] = static_cast<T>(0.5) + (value + tau) / (static_cast<T>(2) * h);
    } else if (value < tau - h) {
        curvature[index] = static_cast<T>(1);
    } else {
        curvature[index] = static_cast<T>(0.5) - (value - tau) / (static_cast<T>(2) * h);
    }
}

template <typename T>
__global__ void weight_score_kernel(T* score, const T* weights, int64_t count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        score[index] *= weights[index];
    }
}

template <typename T>
__global__ void weight_design_kernel(
    const T* design,
    const T* curvature,
    const T* weights,
    T* weighted_design,
    int64_t element_count,
    int64_t columns
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= element_count) {
        return;
    }
    const int64_t row = index / columns;
    T scale = curvature[row];
    if (weights != nullptr) {
        scale *= weights[row];
    }
    weighted_design[index] = design[index] * scale;
}

template <typename T>
__global__ void huber_loss_kernel(
    const T* residual,
    const T* weights,
    T* loss,
    int64_t count,
    T tau
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const T value = residual[index];
    const T absolute = value < static_cast<T>(0) ? -value : value;
    T result = absolute <= tau
        ? static_cast<T>(0.5) * value * value
        : tau * absolute - static_cast<T>(0.5) * tau * tau;
    if (weights != nullptr) {
        result *= weights[index];
    }
    loss[index] = result;
}

template <typename T>
__global__ void subtract_kernel(
    const T* left,
    const T* right,
    T* output,
    int64_t count
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = left[index] - right[index];
    }
}

template <typename T>
__global__ void add_scaled_identity_kernel(
    const T* matrix,
    T* output,
    int64_t side,
    T scale
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t count = side * side;
    if (index >= count) {
        return;
    }
    const int64_t row = index % side;
    const int64_t column = index / side;
    output[index] = matrix[index] + (row == column ? scale : static_cast<T>(0));
}

template <typename T>
__global__ void add_matrix_kernel(
    const T* left,
    const T* right,
    T* output,
    int64_t count
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = left[index] + right[index];
    }
}

template <typename T>
__global__ void scale_and_add_identity_kernel(
    T* matrix,
    int64_t side,
    T scale,
    T ridge
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t count = side * side;
    if (index >= count) {
        return;
    }
    const int64_t row = index % side;
    const int64_t column = index / side;
    matrix[index] = matrix[index] * scale + (row == column ? ridge : static_cast<T>(0));
}

template <typename T>
__global__ void axpby_kernel(
    const T* left,
    T left_scale,
    const T* right,
    T right_scale,
    T* output,
    int64_t count
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = left_scale * left[index] + right_scale * right[index];
    }
}

template <typename T>
__global__ void candidate_kernel(
    const T* beta,
    const T* direction,
    T step,
    T* candidate,
    int64_t count
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        candidate[index] = beta[index] - step * direction[index];
    }
}

template <typename T>
__global__ void pseudoinverse_scale_kernel(
    T* vector,
    const T* singular_values,
    int64_t count,
    T cutoff
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const T singular = singular_values[index];
        vector[index] = singular > cutoff ? vector[index] / singular : static_cast<T>(0);
    }
}

template <typename T>
__global__ void transpose_kernel(
    const T* input,
    T* output,
    int64_t rows,
    int64_t columns
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t count = rows * columns;
    if (index >= count) {
        return;
    }
    const int64_t row = index / columns;
    const int64_t column = index % columns;
    output[column * rows + row] = input[row * columns + column];
}

template <typename T>
cudaError_t last_launch_error() {
    return cudaGetLastError();
}

}  // namespace

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
) {
    residual_score_curvature_kernel<<<blocks_for(count), kThreadsPerBlock, 0, stream>>>(
        residual_input, residual, score, curvature, count, tau, bandwidth
    );
    return last_launch_error<T>();
}

template <typename T>
cudaError_t launch_weight_score(
    T* score,
    const T* weights,
    int64_t count,
    cudaStream_t stream
) {
    weight_score_kernel<<<blocks_for(count), kThreadsPerBlock, 0, stream>>>(score, weights, count);
    return last_launch_error<T>();
}

template <typename T>
cudaError_t launch_weight_design(
    const T* design,
    const T* curvature,
    const T* weights,
    T* weighted_design,
    int64_t rows,
    int64_t columns,
    cudaStream_t stream
) {
    const int64_t count = rows * columns;
    weight_design_kernel<<<blocks_for(count), kThreadsPerBlock, 0, stream>>>(
        design, curvature, weights, weighted_design, count, columns
    );
    return last_launch_error<T>();
}

template <typename T>
cudaError_t launch_huber_loss(
    const T* residual,
    const T* weights,
    T* loss,
    int64_t count,
    T tau,
    cudaStream_t stream
) {
    huber_loss_kernel<<<blocks_for(count), kThreadsPerBlock, 0, stream>>>(
        residual, weights, loss, count, tau
    );
    return last_launch_error<T>();
}

template <typename T>
cudaError_t launch_subtract(
    const T* left,
    const T* right,
    T* output,
    int64_t count,
    cudaStream_t stream
) {
    subtract_kernel<<<blocks_for(count), kThreadsPerBlock, 0, stream>>>(left, right, output, count);
    return last_launch_error<T>();
}

template <typename T>
cudaError_t launch_add_scaled_identity(
    const T* matrix,
    T* output,
    int64_t side,
    T scale,
    cudaStream_t stream
) {
    const int64_t count = side * side;
    add_scaled_identity_kernel<<<blocks_for(count), kThreadsPerBlock, 0, stream>>>(
        matrix, output, side, scale
    );
    return last_launch_error<T>();
}

template <typename T>
cudaError_t launch_add_matrix(
    const T* left,
    const T* right,
    T* output,
    int64_t count,
    cudaStream_t stream
) {
    add_matrix_kernel<<<blocks_for(count), kThreadsPerBlock, 0, stream>>>(left, right, output, count);
    return last_launch_error<T>();
}

template <typename T>
cudaError_t launch_scale_and_add_identity(
    T* matrix,
    int64_t side,
    T scale,
    T ridge,
    cudaStream_t stream
) {
    const int64_t count = side * side;
    scale_and_add_identity_kernel<<<blocks_for(count), kThreadsPerBlock, 0, stream>>>(
        matrix, side, scale, ridge
    );
    return last_launch_error<T>();
}

template <typename T>
cudaError_t launch_axpby(
    const T* left,
    T left_scale,
    const T* right,
    T right_scale,
    T* output,
    int64_t count,
    cudaStream_t stream
) {
    axpby_kernel<<<blocks_for(count), kThreadsPerBlock, 0, stream>>>(
        left, left_scale, right, right_scale, output, count
    );
    return last_launch_error<T>();
}

template <typename T>
cudaError_t launch_candidate(
    const T* beta,
    const T* direction,
    T step,
    T* candidate,
    int64_t count,
    cudaStream_t stream
) {
    candidate_kernel<<<blocks_for(count), kThreadsPerBlock, 0, stream>>>(
        beta, direction, step, candidate, count
    );
    return last_launch_error<T>();
}

template <typename T>
cudaError_t launch_copy(T* destination, const T* source, int64_t count, cudaStream_t stream) {
    return cudaMemcpyAsync(destination, source, sizeof(T) * count, cudaMemcpyDeviceToDevice, stream);
}

template <typename T>
cudaError_t launch_pseudoinverse_scale(
    T* vector,
    const T* singular_values,
    int64_t count,
    T cutoff,
    cudaStream_t stream
) {
    pseudoinverse_scale_kernel<<<blocks_for(count), kThreadsPerBlock, 0, stream>>>(
        vector, singular_values, count, cutoff
    );
    return last_launch_error<T>();
}

template <typename T>
cudaError_t launch_transpose(
    const T* input,
    T* output,
    int64_t rows,
    int64_t columns,
    cudaStream_t stream
) {
    const int64_t count = rows * columns;
    transpose_kernel<<<blocks_for(count), kThreadsPerBlock, 0, stream>>>(
        input, output, rows, columns
    );
    return last_launch_error<T>();
}

template cudaError_t launch_residual_score_curvature<float>(
    const float*, float*, float*, float*, int64_t, float, float, cudaStream_t
);
template cudaError_t launch_residual_score_curvature<double>(
    const double*, double*, double*, double*, int64_t, double, double, cudaStream_t
);
template cudaError_t launch_weight_score<float>(float*, const float*, int64_t, cudaStream_t);
template cudaError_t launch_weight_score<double>(double*, const double*, int64_t, cudaStream_t);
template cudaError_t launch_weight_design<float>(
    const float*, const float*, const float*, float*, int64_t, int64_t, cudaStream_t
);
template cudaError_t launch_weight_design<double>(
    const double*, const double*, const double*, double*, int64_t, int64_t, cudaStream_t
);
template cudaError_t launch_huber_loss<float>(
    const float*, const float*, float*, int64_t, float, cudaStream_t
);
template cudaError_t launch_huber_loss<double>(
    const double*, const double*, double*, int64_t, double, cudaStream_t
);
template cudaError_t launch_subtract<float>(
    const float*, const float*, float*, int64_t, cudaStream_t
);
template cudaError_t launch_subtract<double>(
    const double*, const double*, double*, int64_t, cudaStream_t
);
template cudaError_t launch_add_scaled_identity<float>(
    const float*, float*, int64_t, float, cudaStream_t
);
template cudaError_t launch_add_scaled_identity<double>(
    const double*, double*, int64_t, double, cudaStream_t
);
template cudaError_t launch_add_matrix<float>(
    const float*, const float*, float*, int64_t, cudaStream_t
);
template cudaError_t launch_add_matrix<double>(
    const double*, const double*, double*, int64_t, cudaStream_t
);
template cudaError_t launch_scale_and_add_identity<float>(
    float*, int64_t, float, float, cudaStream_t
);
template cudaError_t launch_scale_and_add_identity<double>(
    double*, int64_t, double, double, cudaStream_t
);
template cudaError_t launch_axpby<float>(
    const float*, float, const float*, float, float*, int64_t, cudaStream_t
);
template cudaError_t launch_axpby<double>(
    const double*, double, const double*, double, double*, int64_t, cudaStream_t
);
template cudaError_t launch_candidate<float>(
    const float*, const float*, float, float*, int64_t, cudaStream_t
);
template cudaError_t launch_candidate<double>(
    const double*, const double*, double, double*, int64_t, cudaStream_t
);
template cudaError_t launch_copy<float>(float*, const float*, int64_t, cudaStream_t);
template cudaError_t launch_copy<double>(double*, const double*, int64_t, cudaStream_t);
template cudaError_t launch_pseudoinverse_scale<float>(
    float*, const float*, int64_t, float, cudaStream_t
);
template cudaError_t launch_pseudoinverse_scale<double>(
    double*, const double*, int64_t, double, cudaStream_t
);
template cudaError_t launch_transpose<float>(
    const float*, float*, int64_t, int64_t, cudaStream_t
);
template cudaError_t launch_transpose<double>(
    const double*, double*, int64_t, int64_t, cudaStream_t
);

}  // namespace rh_cuda
