#include "huber_kernels.cuh"

#include <algorithm>

namespace rh_cuda {
namespace {

constexpr int kThreadsPerBlock = 256;

inline int blocks_for(int64_t count) {
    return static_cast<int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
}

template <typename T>
__global__ void append_intercept_kernel(
    const T* features,
    T* design,
    int64_t rows,
    int64_t feature_columns
) {
    const int64_t design_columns = feature_columns + 1;
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t count = rows * design_columns;
    if (index >= count) {
        return;
    }
    const int64_t row = index / design_columns;
    const int64_t column = index % design_columns;
    design[index] = column == feature_columns
        ? static_cast<T>(1)
        : features[row * feature_columns + column];
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
__global__ void mirror_lower_triangle_kernel(T* matrix, int64_t side) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t count = side * side;
    if (index >= count) {
        return;
    }
    // Matrices in the CUDA engine use column-major layout. cuBLAS SYRKX
    // writes the lower triangle; copy it into the matching upper entries.
    const int64_t row = index % side;
    const int64_t column = index / side;
    if (row < column) {
        matrix[index] = matrix[row * side + column];
    }
}

template <typename T>
__device__ T sign_of(T value) {
    return value > static_cast<T>(0)
        ? static_cast<T>(1)
        : (value < static_cast<T>(0) ? static_cast<T>(-1) : static_cast<T>(0));
}

template <typename T>
__global__ void penalty_sign_kernel(
    const T* coefficients,
    T* output,
    int64_t count,
    int64_t penalized_count
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = index < penalized_count ? sign_of(coefficients[index]) : static_cast<T>(0);
    }
}

template <typename T>
__global__ void weighted_huber_score_kernel(
    const T* residual,
    const T* weights,
    T* score,
    int64_t count,
    T tau
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const T value = residual[index];
    T result = value < -tau ? -tau : (value > tau ? tau : value);
    if (weights != nullptr) {
        result *= weights[index];
    }
    score[index] = result;
}

/*
 * Candidate rounds.  Both kernels run 256 threads per block; the term kernel
 * handles one candidate per grid row and 256 coefficients per block, the
 * residual kernel one batch row per warp.
 */
constexpr int kRoundThreads = 256;
constexpr int kRoundWarps = kRoundThreads / 32;
constexpr int kRoundMaxBlocks = 1024;
constexpr int kRoundRowsPerWarp = 4;
/// Up to this many coefficients a thread computes a whole row's dot products
/// alone. A warp per row would spend its time in the shuffle reduction, which
/// costs a warp-wide add per level and candidate -- ruinous for float64, which
/// this GPU class runs at 1/64 of the float32 rate.
constexpr int64_t kThreadRowMaxParameters = 64;
constexpr unsigned kFullWarp = 0xffffffffu;

template <typename T>
__device__ T form_candidate(
    const CandidateRoundParameters<T>& round,
    int k,
    int64_t index,
    const T* beta,
    const T* direction,
    const T* gradient,
    int64_t penalized_count
) {
    if (round.form == kCandidateNewton) {
        return beta[index] - round.scale[k] * direction[index];
    }
    if (round.form == kCandidateProximal) {
        // Same operation order as the Rust CPU engine: step, then shrink.
        const T value = beta[index] - gradient[index] * round.scale[k];
        const T limit = index < penalized_count ? round.threshold[k] : static_cast<T>(0);
        const T absolute = value < static_cast<T>(0) ? -value : value;
        const T remainder = absolute - limit;
        return sign_of(value) * (remainder > static_cast<T>(0) ? remainder : static_cast<T>(0));
    }
    return beta[index];
}

template <typename T>
__device__ T huber_loss_value(T value, T tau) {
    const T absolute = value < static_cast<T>(0) ? -value : value;
    return absolute <= tau
        ? static_cast<T>(0.5) * value * value
        : tau * absolute - static_cast<T>(0.5) * tau * tau;
}

/// Sum `value` over the block in a fixed order; the total lands in thread 0.
__device__ double block_sum(double value, double* scratch) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(kFullWarp, value, offset);
    }
    if (lane == 0) {
        scratch[warp] = value;
    }
    __syncthreads();
    double total = 0.0;
    if (threadIdx.x == 0) {
        for (int index = 0; index < kRoundWarps; ++index) {
            total += scratch[index];
        }
    }
    __syncthreads();
    return total;
}

/*
 * Form candidate k = blockIdx.y and reduce its coefficient-space terms over
 * this block's 256 coefficients.  The history product (J (c - history))_i is
 * a column-major GEMV row accumulated in T over shared-memory tiles of the
 * delta, like cuBLAS; the five dot products accumulate in double.
 */
template <typename T>
__global__ void candidate_terms_kernel(
    const CandidateRoundParameters<T>* round_pointer,
    const T* beta,
    const T* direction,
    const T* gradient,
    const T* coefficients,
    const T* information,
    const T* penalty_sign,
    T* candidates,
    double* term_partials,
    int64_t parameters,
    int64_t penalized_count
) {
    const CandidateRoundParameters<T> round = *round_pointer;
    const int k = static_cast<int>(blockIdx.y);
    if (k >= round.width) {
        return;
    }
    __shared__ T delta_tile[kRoundThreads];
    __shared__ double scratch[kRoundWarps];
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    T history_product = static_cast<T>(0);
    for (int64_t tile = 0; tile < parameters; tile += kRoundThreads) {
        const int64_t column = tile + threadIdx.x;
        delta_tile[threadIdx.x] = column < parameters
            ? form_candidate(round, k, column, beta, direction, gradient, penalized_count) -
                coefficients[column]
            : static_cast<T>(0);
        __syncthreads();
        if (index < parameters) {
            const int64_t tile_width =
                parameters - tile < kRoundThreads ? parameters - tile : kRoundThreads;
            for (int64_t offset = 0; offset < tile_width; ++offset) {
                history_product += information[index + (tile + offset) * parameters] * delta_tile[offset];
            }
        }
        __syncthreads();
    }

    double terms[kCandidateRoundTermSlots] = {0.0, 0.0, 0.0, 0.0, 0.0};
    if (index < parameters) {
        const T value = form_candidate(round, k, index, beta, direction, gradient, penalized_count);
        candidates[static_cast<int64_t>(k) * parameters + index] = value;
        const T delta = value - coefficients[index];
        const T step = value - beta[index];
        terms[0] = static_cast<double>(delta) * static_cast<double>(history_product);
        terms[1] = static_cast<double>(step) * static_cast<double>(step);
        terms[2] = static_cast<double>(value) * static_cast<double>(value);
        if (gradient != nullptr && round.form != kCandidateCopy) {
            terms[3] = static_cast<double>(gradient[index]) * static_cast<double>(step);
        }
        if (penalty_sign != nullptr) {
            terms[4] = static_cast<double>(penalty_sign[index]) * static_cast<double>(delta);
        }
    }
    double* output = term_partials +
        (static_cast<int64_t>(k) * gridDim.x + blockIdx.x) * kCandidateRoundTermSlots;
    for (int term = 0; term < kCandidateRoundTermSlots; ++term) {
        const double total = block_sum(terms[term], scratch);
        if (threadIdx.x == 0) {
            output[term] = total;
        }
    }
}

/*
 * Residuals and Huber loss of every candidate in the round, one batch row per
 * warp: X is read once for all candidates.  Each block writes its loss
 * partials; the last block to finish folds them, and the term kernel's
 * partials, into `results` in index order and resets the counter.
 */
template <typename T>
__global__ void candidate_residual_loss_kernel(
    const CandidateRoundParameters<T>* round_pointer,
    const T* design,
    const T* y,
    const T* weights,
    const T* candidates,
    T* residuals,
    double* loss_partials,
    const double* term_partials,
    double* results,
    unsigned int* counter,
    int64_t rows,
    int64_t parameters,
    int chunks,
    T tau
) {
    const int width = round_pointer->width;
    __shared__ double scratch[kRoundWarps];
    __shared__ bool last_block;

    double loss[kCandidateRoundWidth] = {0.0, 0.0, 0.0, 0.0};
    // Called by the one thread that owns `row` once its dot products are done.
    const auto finish_row = [&](int64_t row, const T* dot) {
        const T target = y[row];
#pragma unroll
        for (int k = 0; k < kCandidateRoundWidth; ++k) {
            if (k < width) {
                const T residual = target - dot[k];
                residuals[static_cast<int64_t>(k) * rows + row] = residual;
                T value = huber_loss_value(residual, tau);
                if (weights != nullptr) {
                    value *= weights[row];
                }
                // Absolute like the cuBLAS asum this replaces; it only
                // differs for a negative weight, which the estimator
                // rejects but the raw C ABI does not.
                loss[k] += static_cast<double>(value < static_cast<T>(0) ? -value : value);
            }
        }
    };

    // The layout depends on the shape only, never on the round width.
    if (parameters <= kThreadRowMaxParameters) {
        const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
        for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; row < rows;
             row += stride) {
            T dot[kCandidateRoundWidth] = {};
            const T* x = design + row * parameters;
            for (int64_t column = 0; column < parameters; ++column) {
                const T value = x[column];
#pragma unroll
                for (int k = 0; k < kCandidateRoundWidth; ++k) {
                    if (k < width) {
                        dot[k] += value * candidates[static_cast<int64_t>(k) * parameters + column];
                    }
                }
            }
            finish_row(row, dot);
        }
    } else {
        const int lane = threadIdx.x % 32;
        const int warp = threadIdx.x / 32;
        const int64_t warp_stride = static_cast<int64_t>(gridDim.x) * kRoundWarps;
        for (int64_t row = static_cast<int64_t>(blockIdx.x) * kRoundWarps + warp; row < rows;
             row += warp_stride) {
            T dot[kCandidateRoundWidth] = {};
            const T* x = design + row * parameters;
            for (int64_t column = lane; column < parameters; column += 32) {
                const T value = x[column];
#pragma unroll
                for (int k = 0; k < kCandidateRoundWidth; ++k) {
                    if (k < width) {
                        dot[k] += value * candidates[static_cast<int64_t>(k) * parameters + column];
                    }
                }
            }
#pragma unroll
            for (int k = 0; k < kCandidateRoundWidth; ++k) {
                if (k < width) {
                    for (int offset = 16; offset > 0; offset /= 2) {
                        dot[k] += __shfl_down_sync(kFullWarp, dot[k], offset);
                    }
                }
            }
            if (lane == 0) {
                finish_row(row, dot);
            }
        }
    }

    for (int k = 0; k < width; ++k) {
        const double total = block_sum(loss[k], scratch);
        if (threadIdx.x == 0) {
            loss_partials[static_cast<int64_t>(k) * gridDim.x + blockIdx.x] = total;
        }
    }

    // Classic last-block reduction: publish the partials, then count blocks.
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        last_block = atomicAdd(counter, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last_block) {
        return;
    }
    __threadfence();
    for (int k = 0; k < width; ++k) {
        // Thread t folds blocks t, t + 256, ... in order, then a fixed tree.
        double partial = 0.0;
        for (int64_t block = threadIdx.x; block < gridDim.x; block += kRoundThreads) {
            partial += __ldcg(loss_partials + static_cast<int64_t>(k) * gridDim.x + block);
        }
        const double total = block_sum(partial, scratch);
        if (threadIdx.x == 0) {
            results[k * kCandidateRoundSlots] = total;
        }
    }
    if (threadIdx.x < width * kCandidateRoundTermSlots) {
        const int k = threadIdx.x / kCandidateRoundTermSlots;
        const int term = threadIdx.x % kCandidateRoundTermSlots;
        double total = 0.0;
        for (int chunk = 0; chunk < chunks; ++chunk) {
            total += term_partials[(static_cast<int64_t>(k) * chunks + chunk) * kCandidateRoundTermSlots + term];
        }
        results[k * kCandidateRoundSlots + 1 + term] = total;
    }
    if (threadIdx.x == 0) {
        *counter = 0u;
    }
}

template <typename T>
cudaError_t last_launch_error() {
    return cudaGetLastError();
}

}  // namespace

template <typename T>
cudaError_t launch_append_intercept(
    const T* features,
    T* design,
    int64_t rows,
    int64_t feature_columns,
    cudaStream_t stream
) {
    const int64_t count = rows * (feature_columns + 1);
    append_intercept_kernel<<<blocks_for(count), kThreadsPerBlock, 0, stream>>>(
        features, design, rows, feature_columns
    );
    return last_launch_error<T>();
}

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

template <typename T>
cudaError_t launch_mirror_lower_triangle(T* matrix, int64_t side, cudaStream_t stream) {
    const int64_t count = side * side;
    mirror_lower_triangle_kernel<<<blocks_for(count), kThreadsPerBlock, 0, stream>>>(
        matrix, side
    );
    return last_launch_error<T>();
}

template <typename T>
cudaError_t launch_penalty_sign(
    const T* coefficients,
    T* output,
    int64_t count,
    int64_t penalized_count,
    cudaStream_t stream
) {
    penalty_sign_kernel<<<blocks_for(count), kThreadsPerBlock, 0, stream>>>(
        coefficients, output, count, penalized_count
    );
    return last_launch_error<T>();
}

template <typename T>
cudaError_t launch_weighted_huber_score(
    const T* residual,
    const T* weights,
    T* score,
    int64_t count,
    T tau,
    cudaStream_t stream
) {
    weighted_huber_score_kernel<<<blocks_for(count), kThreadsPerBlock, 0, stream>>>(
        residual, weights, score, count, tau
    );
    return last_launch_error<T>();
}

int candidate_round_blocks(int64_t rows, int64_t parameters) {
    const int64_t rows_per_block = parameters <= kThreadRowMaxParameters
        ? static_cast<int64_t>(kRoundThreads)
        : static_cast<int64_t>(kRoundWarps) * kRoundRowsPerWarp;
    const int64_t blocks = (rows + rows_per_block - 1) / rows_per_block;
    return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(blocks, kRoundMaxBlocks)));
}

int candidate_round_chunks(int64_t parameters) {
    return static_cast<int>(std::max<int64_t>(1, (parameters + kRoundThreads - 1) / kRoundThreads));
}

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
) {
    const int chunks = candidate_round_chunks(parameters);
    candidate_terms_kernel<<<dim3(chunks, kCandidateRoundWidth), kRoundThreads, 0, stream>>>(
        buffers.parameters,
        beta,
        direction,
        gradient,
        coefficients,
        information,
        penalty_sign,
        buffers.candidates,
        buffers.term_partials,
        parameters,
        penalized_count
    );
    const cudaError_t terms = cudaGetLastError();
    if (terms != cudaSuccess) {
        return terms;
    }
    candidate_residual_loss_kernel<<<candidate_round_blocks(rows, parameters), kRoundThreads, 0, stream>>>(
        buffers.parameters,
        design,
        y,
        weights,
        buffers.candidates,
        buffers.residuals,
        buffers.loss_partials,
        buffers.term_partials,
        buffers.results,
        buffers.counter,
        rows,
        parameters,
        chunks,
        tau
    );
    return last_launch_error<T>();
}

template cudaError_t launch_append_intercept<float>(
    const float*, float*, int64_t, int64_t, cudaStream_t
);
template cudaError_t launch_append_intercept<double>(
    const double*, double*, int64_t, int64_t, cudaStream_t
);
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
template cudaError_t launch_mirror_lower_triangle<float>(float*, int64_t, cudaStream_t);
template cudaError_t launch_mirror_lower_triangle<double>(double*, int64_t, cudaStream_t);
template cudaError_t launch_penalty_sign<float>(
    const float*, float*, int64_t, int64_t, cudaStream_t
);
template cudaError_t launch_penalty_sign<double>(
    const double*, double*, int64_t, int64_t, cudaStream_t
);
template cudaError_t launch_weighted_huber_score<float>(
    const float*, const float*, float*, int64_t, float, cudaStream_t
);
template cudaError_t launch_weighted_huber_score<double>(
    const double*, const double*, double*, int64_t, double, cudaStream_t
);
template cudaError_t launch_candidate_round<float>(
    const CandidateRoundBuffers<float>&, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, int64_t, int64_t,
    int64_t, float, cudaStream_t
);
template cudaError_t launch_candidate_round<double>(
    const CandidateRoundBuffers<double>&, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*, const double*, const double*,
    int64_t, int64_t, int64_t, double, cudaStream_t
);

}  // namespace rh_cuda
