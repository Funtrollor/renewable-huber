#include "rh_cuda.h"

#include "huber_kernels.cuh"

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using ErrorBuffer = std::array<char, 1024>;
thread_local ErrorBuffer g_last_error{};

void clear_error(ErrorBuffer& buffer) noexcept {
    buffer[0] = '\0';
}

void set_error(ErrorBuffer& buffer, const char* message) noexcept {
    if (message == nullptr) {
        message = "unknown native CUDA failure";
    }
    const size_t length = std::min(std::strlen(message), buffer.size() - 1);
    std::memcpy(buffer.data(), message, length);
    buffer[length] = '\0';
}

class Failure final : public std::exception {
public:
    Failure(RhCudaStatus status, std::string message)
        : status_(status), message_(std::move(message)) {}

    const char* what() const noexcept override { return message_.c_str(); }
    RhCudaStatus status() const noexcept { return status_; }

private:
    RhCudaStatus status_;
    std::string message_;
};

[[noreturn]] void fail(RhCudaStatus status, const std::string& message) {
    throw Failure(status, message);
}

std::string with_code(const char* operation, const char* detail) {
    std::ostringstream stream;
    stream << operation << ": " << detail;
    return stream.str();
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fail(RH_CUDA_STATUS_CUDA_ERROR, with_code(operation, cudaGetErrorString(status)));
    }
}

void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::ostringstream stream;
        stream << operation << ": cuBLAS status " << static_cast<int>(status);
        fail(RH_CUDA_STATUS_CUBLAS_ERROR, stream.str());
    }
}

void check_cusolver(cusolverStatus_t status, const char* operation) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::ostringstream stream;
        stream << operation << ": cuSOLVER status " << static_cast<int>(status);
        fail(RH_CUDA_STATUS_CUSOLVER_ERROR, stream.str());
    }
}

template <typename Struct>
void check_header(const Struct* value, const char* name) {
    if (value == nullptr) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, std::string(name) + " must not be null");
    }
    if (value->abi_version != RH_CUDA_ABI_VERSION || value->struct_size < sizeof(Struct)) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, std::string(name) + " has an unsupported ABI header");
    }
}

bool finite_positive(double value) {
    return std::isfinite(value) && value > 0.0;
}

bool finite_nonnegative(double value) {
    return std::isfinite(value) && value >= 0.0;
}

size_t checked_elements(int64_t rows, int64_t columns, const char* name) {
    if (rows < 0 || columns < 0) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, std::string(name) + " has a negative shape");
    }
    const uint64_t left = static_cast<uint64_t>(rows);
    const uint64_t right = static_cast<uint64_t>(columns);
    if (right != 0 && left > std::numeric_limits<size_t>::max() / right) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, std::string(name) + " shape overflows address space");
    }
    return static_cast<size_t>(left * right);
}

template <typename T>
void validate_finite_state(
    const T* coefficients,
    const T* information,
    size_t parameters
) {
    for (size_t index = 0; index < parameters; ++index) {
        if (!std::isfinite(static_cast<double>(coefficients[index]))) {
            fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "restored coefficients must be finite");
        }
    }

    for (size_t row = 0; row < parameters; ++row) {
        for (size_t column = 0; column < parameters; ++column) {
            const T value = information[row * parameters + column];
            if (!std::isfinite(static_cast<double>(value))) {
                fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "restored information must be finite");
            }
        }
    }
}

template <typename T>
void allocate(void** pointer, size_t elements, const char* name) {
    if (elements == 0) {
        *pointer = nullptr;
        return;
    }
    if (elements > std::numeric_limits<size_t>::max() / sizeof(T)) {
        fail(RH_CUDA_STATUS_ALLOCATION_FAILED, std::string(name) + " allocation overflows");
    }
    const cudaError_t status = cudaMalloc(pointer, elements * sizeof(T));
    if (status != cudaSuccess) {
        fail(RH_CUDA_STATUS_ALLOCATION_FAILED, with_code(name, cudaGetErrorString(status)));
    }
}

void release(void*& pointer) noexcept {
    if (pointer != nullptr) {
        cudaFree(pointer);
        pointer = nullptr;
    }
}

template <typename T>
struct Blas;

template <>
struct Blas<float> {
    static cublasStatus_t copy(cublasHandle_t h, int n, const float* x, float* y) {
        return cublasScopy(h, n, x, 1, y, 1);
    }
    static cublasStatus_t gemv(
        cublasHandle_t h,
        cublasOperation_t op,
        int m,
        int n,
        const float* alpha,
        const float* a,
        int lda,
        const float* x,
        const float* beta,
        float* y
    ) {
        return cublasSgemv(h, op, m, n, alpha, a, lda, x, 1, beta, y, 1);
    }
    static cublasStatus_t gemm(
        cublasHandle_t h,
        cublasOperation_t op_a,
        cublasOperation_t op_b,
        int m,
        int n,
        int k,
        const float* alpha,
        const float* a,
        int lda,
        const float* b,
        int ldb,
        const float* beta,
        float* c,
        int ldc
    ) {
        return cublasSgemm(h, op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    static cublasStatus_t asum(cublasHandle_t h, int n, const float* x, float* result) {
        return cublasSasum(h, n, x, 1, result);
    }
    static cublasStatus_t dot(cublasHandle_t h, int n, const float* x, const float* y, float* result) {
        return cublasSdot(h, n, x, 1, y, 1, result);
    }
    static cublasStatus_t nrm2(cublasHandle_t h, int n, const float* x, float* result) {
        return cublasSnrm2(h, n, x, 1, result);
    }
};

template <>
struct Blas<double> {
    static cublasStatus_t copy(cublasHandle_t h, int n, const double* x, double* y) {
        return cublasDcopy(h, n, x, 1, y, 1);
    }
    static cublasStatus_t gemv(
        cublasHandle_t h,
        cublasOperation_t op,
        int m,
        int n,
        const double* alpha,
        const double* a,
        int lda,
        const double* x,
        const double* beta,
        double* y
    ) {
        return cublasDgemv(h, op, m, n, alpha, a, lda, x, 1, beta, y, 1);
    }
    static cublasStatus_t gemm(
        cublasHandle_t h,
        cublasOperation_t op_a,
        cublasOperation_t op_b,
        int m,
        int n,
        int k,
        const double* alpha,
        const double* a,
        int lda,
        const double* b,
        int ldb,
        const double* beta,
        double* c,
        int ldc
    ) {
        return cublasDgemm(h, op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    static cublasStatus_t asum(cublasHandle_t h, int n, const double* x, double* result) {
        return cublasDasum(h, n, x, 1, result);
    }
    static cublasStatus_t dot(
        cublasHandle_t h,
        int n,
        const double* x,
        const double* y,
        double* result
    ) {
        return cublasDdot(h, n, x, 1, y, 1, result);
    }
    static cublasStatus_t nrm2(cublasHandle_t h, int n, const double* x, double* result) {
        return cublasDnrm2(h, n, x, 1, result);
    }
};

template <typename T>
struct Solver;

template <>
struct Solver<float> {
    static cusolverStatus_t getrf_buffer_size(
        cusolverDnHandle_t h, int n, float* a, int lda, int* lwork
    ) {
        return cusolverDnSgetrf_bufferSize(h, n, n, a, lda, lwork);
    }
    static cusolverStatus_t getrf(
        cusolverDnHandle_t h, int n, float* a, int lda, float* work, int* pivots, int* info
    ) {
        return cusolverDnSgetrf(h, n, n, a, lda, work, pivots, info);
    }
    static cusolverStatus_t getrs(
        cusolverDnHandle_t h,
        int n,
        const float* a,
        int lda,
        const int* pivots,
        float* rhs,
        int ldb,
        int* info
    ) {
        return cusolverDnSgetrs(h, CUBLAS_OP_N, n, 1, a, lda, pivots, rhs, ldb, info);
    }
    static cusolverStatus_t gesvdj_buffer_size(
        cusolverDnHandle_t h,
        float* a,
        float* singular,
        float* u,
        float* v,
        int n,
        int* lwork,
        gesvdjInfo_t params
    ) {
        return cusolverDnSgesvdj_bufferSize(
            h, CUSOLVER_EIG_MODE_VECTOR, 0, n, n, a, n, singular, u, n, v, n, lwork, params
        );
    }
    static cusolverStatus_t gesvdj(
        cusolverDnHandle_t h,
        float* a,
        float* singular,
        float* u,
        float* v,
        int n,
        float* work,
        int lwork,
        int* info,
        gesvdjInfo_t params
    ) {
        return cusolverDnSgesvdj(
            h, CUSOLVER_EIG_MODE_VECTOR, 0, n, n, a, n, singular, u, n, v, n, work, lwork, info,
            params
        );
    }
};

template <>
struct Solver<double> {
    static cusolverStatus_t getrf_buffer_size(
        cusolverDnHandle_t h, int n, double* a, int lda, int* lwork
    ) {
        return cusolverDnDgetrf_bufferSize(h, n, n, a, lda, lwork);
    }
    static cusolverStatus_t getrf(
        cusolverDnHandle_t h, int n, double* a, int lda, double* work, int* pivots, int* info
    ) {
        return cusolverDnDgetrf(h, n, n, a, lda, work, pivots, info);
    }
    static cusolverStatus_t getrs(
        cusolverDnHandle_t h,
        int n,
        const double* a,
        int lda,
        const int* pivots,
        double* rhs,
        int ldb,
        int* info
    ) {
        return cusolverDnDgetrs(h, CUBLAS_OP_N, n, 1, a, lda, pivots, rhs, ldb, info);
    }
    static cusolverStatus_t gesvdj_buffer_size(
        cusolverDnHandle_t h,
        double* a,
        double* singular,
        double* u,
        double* v,
        int n,
        int* lwork,
        gesvdjInfo_t params
    ) {
        return cusolverDnDgesvdj_bufferSize(
            h, CUSOLVER_EIG_MODE_VECTOR, 0, n, n, a, n, singular, u, n, v, n, lwork, params
        );
    }
    static cusolverStatus_t gesvdj(
        cusolverDnHandle_t h,
        double* a,
        double* singular,
        double* u,
        double* v,
        int n,
        double* work,
        int lwork,
        int* info,
        gesvdjInfo_t params
    ) {
        return cusolverDnDgesvdj(
            h, CUSOLVER_EIG_MODE_VECTOR, 0, n, n, a, n, singular, u, n, v, n, work, lwork, info,
            params
        );
    }
};

}  // namespace

struct RhCudaEngine {
    RhCudaDType dtype = RH_CUDA_DTYPE_FLOAT64;
    int32_t device_id = 0;
    int64_t n_parameters = 0;
    int64_t capacity_rows = 0;

    int64_t n_samples_seen = 0;
    int64_t batch_count = 0;
    double previous_lambda = 0.0;
    double weight_sum = 0.0;

    cudaStream_t stream = nullptr;
    cublasHandle_t cublas = nullptr;
    cusolverDnHandle_t solver = nullptr;
    gesvdjInfo_t svd_params = nullptr;

    void* d_coefficients = nullptr;
    void* d_information = nullptr;
    void* d_information_next = nullptr;
    void* d_trial_beta = nullptr;
    void* d_candidate = nullptr;
    void* d_delta = nullptr;
    void* d_history_vector = nullptr;
    void* d_gradient = nullptr;
    void* d_direction = nullptr;
    void* d_gram = nullptr;
    void* d_hessian = nullptr;
    void* d_factor = nullptr;
    void* d_singular_values = nullptr;
    void* d_svd_u = nullptr;
    void* d_svd_v = nullptr;
    void* d_svd_vector = nullptr;
    void* d_lu_work = nullptr;
    void* d_svd_work = nullptr;
    int* d_pivots = nullptr;
    int* d_solver_info = nullptr;
    int lu_lwork = 0;
    int svd_lwork = 0;

    void* d_design = nullptr;
    void* d_y = nullptr;
    void* d_weights = nullptr;
    void* d_residual = nullptr;
    void* d_score = nullptr;
    void* d_curvature = nullptr;
    void* d_loss = nullptr;
    void* d_weighted_design = nullptr;

    ErrorBuffer last_error{};

    ~RhCudaEngine() noexcept {
        if (device_id >= 0) {
            cudaSetDevice(device_id);
        }
        release(d_coefficients);
        release(d_information);
        release(d_information_next);
        release(d_trial_beta);
        release(d_candidate);
        release(d_delta);
        release(d_history_vector);
        release(d_gradient);
        release(d_direction);
        release(d_gram);
        release(d_hessian);
        release(d_factor);
        release(d_singular_values);
        release(d_svd_u);
        release(d_svd_v);
        release(d_svd_vector);
        release(d_lu_work);
        release(d_svd_work);
        void* pivots = d_pivots;
        release(pivots);
        d_pivots = nullptr;
        void* solver_info = d_solver_info;
        release(solver_info);
        d_solver_info = nullptr;
        release(d_design);
        release(d_y);
        release(d_weights);
        release(d_residual);
        release(d_score);
        release(d_curvature);
        release(d_loss);
        release(d_weighted_design);
        if (svd_params != nullptr) {
            cusolverDnDestroyGesvdjInfo(svd_params);
        }
        if (solver != nullptr) {
            cusolverDnDestroy(solver);
        }
        if (cublas != nullptr) {
            cublasDestroy(cublas);
        }
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }
};

namespace {

void set_device(const RhCudaEngine* engine) {
    if (engine == nullptr) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "engine must not be null");
    }
    check_cuda(cudaSetDevice(engine->device_id), "cudaSetDevice");
}

void validate_options(const RhCudaEngineOptions* options) {
    check_header(options, "engine options");
    if (options->dtype != RH_CUDA_DTYPE_FLOAT32 && options->dtype != RH_CUDA_DTYPE_FLOAT64) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "engine dtype must be float32 or float64");
    }
    if (options->device_id < 0) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "device_id must be non-negative");
    }
    if (options->n_parameters <= 0 || options->n_parameters > std::numeric_limits<int>::max()) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "n_parameters is outside the CUDA dense-solver range");
    }
}

template <typename T>
T* typed(void* value) {
    return static_cast<T*>(value);
}

template <typename T>
const T* typed(const void* value) {
    return static_cast<const T*>(value);
}

template <typename T>
void allocate_static_buffers(RhCudaEngine* engine) {
    const size_t p = static_cast<size_t>(engine->n_parameters);
    const size_t square = checked_elements(engine->n_parameters, engine->n_parameters, "state matrix");

    allocate<T>(&engine->d_coefficients, p, "coefficients");
    allocate<T>(&engine->d_information, square, "information");
    allocate<T>(&engine->d_information_next, square, "next information");
    allocate<T>(&engine->d_trial_beta, p, "trial coefficients");
    allocate<T>(&engine->d_candidate, p, "candidate coefficients");
    allocate<T>(&engine->d_delta, p, "coefficient delta");
    allocate<T>(&engine->d_history_vector, p, "history vector");
    allocate<T>(&engine->d_gradient, p, "gradient");
    allocate<T>(&engine->d_direction, p, "Newton direction");
    allocate<T>(&engine->d_gram, square, "weighted gram");
    allocate<T>(&engine->d_hessian, square, "Hessian");
    allocate<T>(&engine->d_factor, square, "factor matrix");
    allocate<T>(&engine->d_singular_values, p, "singular values");
    allocate<T>(&engine->d_svd_u, square, "SVD U");
    allocate<T>(&engine->d_svd_v, square, "SVD V");
    allocate<T>(&engine->d_svd_vector, p, "SVD vector");
    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&engine->d_pivots), p * sizeof(int)),
        "LU pivots"
    );
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&engine->d_solver_info), sizeof(int)), "solver info");

    check_cusolver(cusolverDnCreateGesvdjInfo(&engine->svd_params), "cusolverDnCreateGesvdjInfo");
    check_cusolver(
        cusolverDnXgesvdjSetTolerance(engine->svd_params, static_cast<double>(std::numeric_limits<T>::epsilon())),
        "cusolverDnXgesvdjSetTolerance"
    );
    check_cusolver(cusolverDnXgesvdjSetMaxSweeps(engine->svd_params, 100), "cusolverDnXgesvdjSetMaxSweeps");

    const int n = static_cast<int>(engine->n_parameters);
    check_cusolver(
        Solver<T>::getrf_buffer_size(engine->solver, n, typed<T>(engine->d_factor), n, &engine->lu_lwork),
        "cusolverDn*getrf_bufferSize"
    );
    check_cusolver(
        Solver<T>::gesvdj_buffer_size(
            engine->solver,
            typed<T>(engine->d_factor),
            typed<T>(engine->d_singular_values),
            typed<T>(engine->d_svd_u),
            typed<T>(engine->d_svd_v),
            n,
            &engine->svd_lwork,
            engine->svd_params
        ),
        "cusolverDn*gesvdj_bufferSize"
    );
    allocate<T>(&engine->d_lu_work, static_cast<size_t>(engine->lu_lwork), "LU workspace");
    allocate<T>(&engine->d_svd_work, static_cast<size_t>(engine->svd_lwork), "SVD workspace");

    check_cuda(cudaMemsetAsync(engine->d_coefficients, 0, p * sizeof(T), engine->stream), "zero coefficients");
    check_cuda(cudaMemsetAsync(engine->d_information, 0, square * sizeof(T), engine->stream), "zero information");
    check_cuda(cudaStreamSynchronize(engine->stream), "initial state synchronization");
}

void release_batch_buffers(RhCudaEngine* engine) noexcept {
    release(engine->d_design);
    release(engine->d_y);
    release(engine->d_weights);
    release(engine->d_residual);
    release(engine->d_score);
    release(engine->d_curvature);
    release(engine->d_loss);
    release(engine->d_weighted_design);
    engine->capacity_rows = 0;
}

template <typename T>
void ensure_batch_capacity(RhCudaEngine* engine, int64_t rows) {
    if (rows <= engine->capacity_rows) {
        return;
    }
    release_batch_buffers(engine);
    const size_t vector = static_cast<size_t>(rows);
    const size_t matrix = checked_elements(rows, engine->n_parameters, "batch design");
    allocate<T>(&engine->d_design, matrix, "batch design");
    allocate<T>(&engine->d_y, vector, "batch target");
    allocate<T>(&engine->d_weights, vector, "batch weights");
    allocate<T>(&engine->d_residual, vector, "residual");
    allocate<T>(&engine->d_score, vector, "score");
    allocate<T>(&engine->d_curvature, vector, "curvature");
    allocate<T>(&engine->d_loss, vector, "loss");
    allocate<T>(&engine->d_weighted_design, matrix, "weighted design");
    engine->capacity_rows = rows;
}

void validate_config(const RhCudaUnpenalizedConfig* config, const RhCudaEngine* engine) {
    check_header(config, "unpenalized config");
    if (config->n_features_in <= 0 || config->n_features_in > engine->n_parameters) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "n_features_in is incompatible with engine state");
    }
    if (config->max_iter < 1 || config->max_iter > std::numeric_limits<int>::max()) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "max_iter must be in the CUDA solver range");
    }
    if (!finite_positive(config->tau) || !finite_positive(config->bandwidth_scale) ||
        !finite_positive(config->tolerance) || !finite_nonnegative(config->ridge)) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "unpenalized config contains an invalid numerical value");
    }
}

void validate_batch(const RhCudaHostBatch* batch, const RhCudaEngine* engine) {
    check_header(batch, "host batch");
    if (batch->x_design == nullptr || batch->y == nullptr) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "x_design and y must not be null");
    }
    if (batch->n_rows <= 0 || batch->n_rows > std::numeric_limits<int>::max()) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "n_rows must be in the CUDA BLAS range");
    }
    if (batch->n_columns != engine->n_parameters) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "x_design column count does not match engine n_parameters");
    }
    if (!finite_positive(batch->batch_weight)) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "batch_weight must be finite and positive");
    }
}

double bandwidth_for(const RhCudaEngine* engine, const RhCudaHostBatch* batch, const RhCudaUnpenalizedConfig* config) {
    const double n_total = engine->weight_sum + batch->batch_weight;
    if (!finite_positive(n_total)) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "cumulative sample weight must be finite and positive");
    }
    const double predictors = static_cast<double>(std::max<int64_t>(config->n_features_in, 2));
    const double raw = config->bandwidth_scale / (std::sqrt(n_total) * std::log(predictors));
    return std::min(raw, config->tau);
}

template <typename T>
void copy_host_batch(RhCudaEngine* engine, const RhCudaHostBatch* batch) {
    ensure_batch_capacity<T>(engine, batch->n_rows);
    const size_t matrix = checked_elements(batch->n_rows, engine->n_parameters, "host batch design");
    const size_t vector = static_cast<size_t>(batch->n_rows);
    check_cuda(
        cudaMemcpyAsync(engine->d_design, batch->x_design, matrix * sizeof(T), cudaMemcpyHostToDevice, engine->stream),
        "copy X_design to device"
    );
    check_cuda(
        cudaMemcpyAsync(engine->d_y, batch->y, vector * sizeof(T), cudaMemcpyHostToDevice, engine->stream),
        "copy y to device"
    );
    if (batch->sample_weight != nullptr) {
        check_cuda(
            cudaMemcpyAsync(
                engine->d_weights, batch->sample_weight, vector * sizeof(T), cudaMemcpyHostToDevice, engine->stream
            ),
            "copy sample_weight to device"
        );
    }
}

template <typename T>
void compute_residual(RhCudaEngine* engine, int rows, const T* beta) {
    const T negative_one = static_cast<T>(-1);
    const T one = static_cast<T>(1);
    const int parameters = static_cast<int>(engine->n_parameters);
    check_cublas(
        Blas<T>::copy(engine->cublas, rows, typed<T>(engine->d_y), typed<T>(engine->d_residual)),
        "copy y into residual"
    );
    /*
     * X_design arrives C-row-major (rows, parameters).  Its byte layout is a
     * column-major (parameters, rows) X^T, so GEMV with op=T computes X beta
     * without a transpose or an extra device allocation.
     */
    check_cublas(
        Blas<T>::gemv(
            engine->cublas,
            CUBLAS_OP_T,
            parameters,
            rows,
            &negative_one,
            typed<T>(engine->d_design),
            parameters,
            beta,
            &one,
            typed<T>(engine->d_residual)
        ),
        "compute residual"
    );
}

template <typename T>
double smooth_objective(
    RhCudaEngine* engine,
    int rows,
    const T* beta,
    const T* weights,
    T tau,
    double n_total
) {
    const int parameters = static_cast<int>(engine->n_parameters);
    compute_residual<T>(engine, rows, beta);
    check_cuda(
        rh_cuda::launch_huber_loss(
            typed<T>(engine->d_residual), weights, typed<T>(engine->d_loss), rows, tau, engine->stream
        ),
        "launch Huber loss"
    );
    T current_loss = static_cast<T>(0);
    check_cublas(
        Blas<T>::asum(engine->cublas, rows, typed<T>(engine->d_loss), &current_loss),
        "reduce Huber loss"
    );

    check_cuda(
        rh_cuda::launch_subtract(
            beta, typed<T>(engine->d_coefficients), typed<T>(engine->d_delta), parameters, engine->stream
        ),
        "form historical coefficient delta"
    );
    const T one = static_cast<T>(1);
    const T zero = static_cast<T>(0);
    check_cublas(
        Blas<T>::gemv(
            engine->cublas,
            CUBLAS_OP_N,
            parameters,
            parameters,
            &one,
            typed<T>(engine->d_information),
            parameters,
            typed<T>(engine->d_delta),
            &zero,
            typed<T>(engine->d_history_vector)
        ),
        "compute historical objective term"
    );
    T historical = static_cast<T>(0);
    check_cublas(
        Blas<T>::dot(
            engine->cublas,
            parameters,
            typed<T>(engine->d_delta),
            typed<T>(engine->d_history_vector),
            &historical
        ),
        "reduce historical objective term"
    );
    return (static_cast<double>(current_loss) + 0.5 * static_cast<double>(historical)) / n_total;
}

template <typename T>
void compute_gradient_hessian(
    RhCudaEngine* engine,
    int rows,
    const T* beta,
    const T* weights,
    T tau,
    T bandwidth,
    double n_total,
    T ridge
) {
    const int parameters = static_cast<int>(engine->n_parameters);
    compute_residual<T>(engine, rows, beta);
    check_cuda(
        rh_cuda::launch_residual_score_curvature(
            typed<T>(engine->d_residual),
            typed<T>(engine->d_residual),
            typed<T>(engine->d_score),
            typed<T>(engine->d_curvature),
            rows,
            tau,
            bandwidth,
            engine->stream
        ),
        "launch residual, Huber score, and curvature"
    );
    if (weights != nullptr) {
        check_cuda(
            rh_cuda::launch_weight_score(typed<T>(engine->d_score), weights, rows, engine->stream),
            "apply sample weight to score"
        );
    }

    const T negative_inv_total = static_cast<T>(-1.0 / n_total);
    const T positive_inv_total = static_cast<T>(1.0 / n_total);
    const T one = static_cast<T>(1);
    const T zero = static_cast<T>(0);
    check_cublas(
        Blas<T>::gemv(
            engine->cublas,
            CUBLAS_OP_N,
            parameters,
            rows,
            &negative_inv_total,
            typed<T>(engine->d_design),
            parameters,
            typed<T>(engine->d_score),
            &zero,
            typed<T>(engine->d_gradient)
        ),
        "compute current gradient"
    );
    check_cuda(
        rh_cuda::launch_subtract(
            beta, typed<T>(engine->d_coefficients), typed<T>(engine->d_delta), parameters, engine->stream
        ),
        "form gradient coefficient delta"
    );
    check_cublas(
        Blas<T>::gemv(
            engine->cublas,
            CUBLAS_OP_N,
            parameters,
            parameters,
            &positive_inv_total,
            typed<T>(engine->d_information),
            parameters,
            typed<T>(engine->d_delta),
            &one,
            typed<T>(engine->d_gradient)
        ),
        "add historical gradient"
    );

    check_cuda(
        rh_cuda::launch_weight_design(
            typed<T>(engine->d_design),
            typed<T>(engine->d_curvature),
            weights,
            typed<T>(engine->d_weighted_design),
            rows,
            parameters,
            engine->stream
        ),
        "form weighted design"
    );
    /* X^T (W X), using row-major buffers interpreted as column-major transposes. */
    check_cublas(
        Blas<T>::gemm(
            engine->cublas,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            parameters,
            parameters,
            rows,
            &one,
            typed<T>(engine->d_design),
            parameters,
            typed<T>(engine->d_weighted_design),
            parameters,
            &zero,
            typed<T>(engine->d_gram),
            parameters
        ),
        "compute weighted Gram matrix"
    );
    const int64_t square = engine->n_parameters * engine->n_parameters;
    check_cuda(
        rh_cuda::launch_add_matrix(
            typed<T>(engine->d_gram),
            typed<T>(engine->d_information),
            typed<T>(engine->d_hessian),
            square,
            engine->stream
        ),
        "add historical information to Hessian"
    );
    check_cuda(
        rh_cuda::launch_scale_and_add_identity(
            typed<T>(engine->d_hessian),
            parameters,
            positive_inv_total,
            ridge,
            engine->stream
        ),
        "scale Hessian and add ridge"
    );
}

template <typename T>
void solve_minimum_norm_svd(RhCudaEngine* engine, bool* used_fallback) {
    const int parameters = static_cast<int>(engine->n_parameters);
    const size_t square = checked_elements(engine->n_parameters, engine->n_parameters, "SVD factor");
    check_cuda(
        cudaMemcpyAsync(
            engine->d_factor,
            engine->d_hessian,
            square * sizeof(T),
            cudaMemcpyDeviceToDevice,
            engine->stream
        ),
        "copy Hessian for SVD fallback"
    );
    check_cusolver(
        Solver<T>::gesvdj(
            engine->solver,
            typed<T>(engine->d_factor),
            typed<T>(engine->d_singular_values),
            typed<T>(engine->d_svd_u),
            typed<T>(engine->d_svd_v),
            parameters,
            typed<T>(engine->d_svd_work),
            engine->svd_lwork,
            engine->d_solver_info,
            engine->svd_params
        ),
        "cusolverDn*gesvdj"
    );
    int info = 0;
    check_cuda(
        cudaMemcpyAsync(&info, engine->d_solver_info, sizeof(int), cudaMemcpyDeviceToHost, engine->stream),
        "read SVD solver info"
    );
    check_cuda(cudaStreamSynchronize(engine->stream), "wait for SVD fallback");
    if (info != 0) {
        fail(RH_CUDA_STATUS_NUMERICAL_ERROR, "minimum-norm SVD fallback did not converge");
    }

    std::vector<T> singular(static_cast<size_t>(parameters));
    check_cuda(
        cudaMemcpyAsync(
            singular.data(),
            engine->d_singular_values,
            singular.size() * sizeof(T),
            cudaMemcpyDeviceToHost,
            engine->stream
        ),
        "copy SVD singular values"
    );
    check_cuda(cudaStreamSynchronize(engine->stream), "read SVD singular values");
    const T maximum = *std::max_element(singular.begin(), singular.end());
    if (!std::isfinite(static_cast<double>(maximum))) {
        fail(RH_CUDA_STATUS_NUMERICAL_ERROR, "SVD fallback produced non-finite singular values");
    }
    const T cutoff = static_cast<T>(
        std::numeric_limits<T>::epsilon() * static_cast<T>(parameters) * maximum
    );

    const T one = static_cast<T>(1);
    const T zero = static_cast<T>(0);
    check_cublas(
        Blas<T>::gemv(
            engine->cublas,
            CUBLAS_OP_T,
            parameters,
            parameters,
            &one,
            typed<T>(engine->d_svd_u),
            parameters,
            typed<T>(engine->d_gradient),
            &zero,
            typed<T>(engine->d_svd_vector)
        ),
        "project gradient into SVD basis"
    );
    check_cuda(
        rh_cuda::launch_pseudoinverse_scale(
            typed<T>(engine->d_svd_vector),
            typed<T>(engine->d_singular_values),
            parameters,
            cutoff,
            engine->stream
        ),
        "apply SVD pseudo-inverse cutoff"
    );
    check_cublas(
        Blas<T>::gemv(
            engine->cublas,
            CUBLAS_OP_N,
            parameters,
            parameters,
            &one,
            typed<T>(engine->d_svd_v),
            parameters,
            typed<T>(engine->d_svd_vector),
            &zero,
            typed<T>(engine->d_direction)
        ),
        "form minimum-norm Newton direction"
    );
    *used_fallback = true;
}

template <typename T>
void solve_direction(RhCudaEngine* engine, bool* used_fallback) {
    const int parameters = static_cast<int>(engine->n_parameters);
    const size_t square = checked_elements(engine->n_parameters, engine->n_parameters, "LU factor");
    check_cuda(
        cudaMemcpyAsync(
            engine->d_factor,
            engine->d_hessian,
            square * sizeof(T),
            cudaMemcpyDeviceToDevice,
            engine->stream
        ),
        "copy Hessian for LU"
    );
    check_cusolver(
        Solver<T>::getrf(
            engine->solver,
            parameters,
            typed<T>(engine->d_factor),
            parameters,
            typed<T>(engine->d_lu_work),
            engine->d_pivots,
            engine->d_solver_info
        ),
        "cusolverDn*getrf"
    );
    int info = 0;
    check_cuda(
        cudaMemcpyAsync(&info, engine->d_solver_info, sizeof(int), cudaMemcpyDeviceToHost, engine->stream),
        "read LU factorization info"
    );
    check_cuda(cudaStreamSynchronize(engine->stream), "wait for LU factorization");
    if (info < 0) {
        fail(RH_CUDA_STATUS_INTERNAL_ERROR, "LU factorization received an invalid cuSOLVER argument");
    }
    if (info > 0) {
        solve_minimum_norm_svd<T>(engine, used_fallback);
        return;
    }
    check_cublas(
        Blas<T>::copy(
            engine->cublas, parameters, typed<T>(engine->d_gradient), typed<T>(engine->d_direction)
        ),
        "copy gradient into solve RHS"
    );
    check_cusolver(
        Solver<T>::getrs(
            engine->solver,
            parameters,
            typed<T>(engine->d_factor),
            parameters,
            engine->d_pivots,
            typed<T>(engine->d_direction),
            parameters,
            engine->d_solver_info
        ),
        "cusolverDn*getrs"
    );
    check_cuda(
        cudaMemcpyAsync(&info, engine->d_solver_info, sizeof(int), cudaMemcpyDeviceToHost, engine->stream),
        "read LU solve info"
    );
    check_cuda(cudaStreamSynchronize(engine->stream), "wait for LU solve");
    if (info != 0) {
        fail(
            RH_CUDA_STATUS_NUMERICAL_ERROR,
            info < 0 ? "LU solve received an invalid cuSOLVER argument"
                     : "LU solve failed to produce a finite direction"
        );
    }
}

template <typename T>
void final_information(
    RhCudaEngine* engine,
    int rows,
    const T* weights,
    T tau,
    T bandwidth
) {
    const int parameters = static_cast<int>(engine->n_parameters);
    const T one = static_cast<T>(1);
    const T zero = static_cast<T>(0);
    compute_residual<T>(engine, rows, typed<T>(engine->d_trial_beta));
    check_cuda(
        rh_cuda::launch_residual_score_curvature(
            typed<T>(engine->d_residual),
            typed<T>(engine->d_residual),
            typed<T>(engine->d_score),
            typed<T>(engine->d_curvature),
            rows,
            tau,
            bandwidth,
            engine->stream
        ),
        "launch final curvature"
    );
    check_cuda(
        rh_cuda::launch_weight_design(
            typed<T>(engine->d_design),
            typed<T>(engine->d_curvature),
            weights,
            typed<T>(engine->d_weighted_design),
            rows,
            parameters,
            engine->stream
        ),
        "form final weighted design"
    );
    check_cublas(
        Blas<T>::gemm(
            engine->cublas,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            parameters,
            parameters,
            rows,
            &one,
            typed<T>(engine->d_design),
            parameters,
            typed<T>(engine->d_weighted_design),
            parameters,
            &zero,
            typed<T>(engine->d_gram),
            parameters
        ),
        "compute final information increment"
    );
    check_cuda(
        rh_cuda::launch_add_matrix(
            typed<T>(engine->d_information),
            typed<T>(engine->d_gram),
            typed<T>(engine->d_information_next),
            engine->n_parameters * engine->n_parameters,
            engine->stream
        ),
        "form next renewable information"
    );
}

struct SolveOutcome {
    int64_t iterations = 0;
    bool converged = false;
    double objective = 0.0;
    bool used_fallback = false;
};

template <typename T>
SolveOutcome solve_unpenalized(
    RhCudaEngine* engine,
    int rows,
    const T* weights,
    const RhCudaUnpenalizedConfig* config,
    double n_total,
    T tau,
    T bandwidth
) {
    const int parameters = static_cast<int>(engine->n_parameters);
    check_cuda(
        rh_cuda::launch_copy(
            typed<T>(engine->d_trial_beta), typed<T>(engine->d_coefficients), parameters, engine->stream
        ),
        "initialize Newton coefficients"
    );

    SolveOutcome outcome;
    outcome.objective = smooth_objective<T>(
        engine, rows, typed<T>(engine->d_trial_beta), weights, tau, n_total
    );

    for (int iteration = 1; iteration <= static_cast<int>(config->max_iter); ++iteration) {
        compute_gradient_hessian<T>(
            engine,
            rows,
            typed<T>(engine->d_trial_beta),
            weights,
            tau,
            bandwidth,
            n_total,
            static_cast<T>(config->ridge)
        );
        solve_direction<T>(engine, &outcome.used_fallback);

        bool accepted = false;
        double candidate_objective = outcome.objective;
        for (int backtrack = 0; backtrack <= 26; ++backtrack) {
            const T step = std::ldexp(static_cast<T>(1), -backtrack);
            check_cuda(
                rh_cuda::launch_candidate(
                    typed<T>(engine->d_trial_beta),
                    typed<T>(engine->d_direction),
                    step,
                    typed<T>(engine->d_candidate),
                    parameters,
                    engine->stream
                ),
                "form line-search candidate"
            );
            candidate_objective = smooth_objective<T>(
                engine, rows, typed<T>(engine->d_candidate), weights, tau, n_total
            );
            if (candidate_objective <= outcome.objective) {
                accepted = true;
                break;
            }
        }
        if (!accepted) {
            outcome.iterations = iteration;
            outcome.converged = false;
            return outcome;
        }

        check_cuda(
            rh_cuda::launch_subtract(
                typed<T>(engine->d_candidate),
                typed<T>(engine->d_trial_beta),
                typed<T>(engine->d_delta),
                parameters,
                engine->stream
            ),
            "form Newton coefficient difference"
        );
        T difference_norm = static_cast<T>(0);
        T beta_norm = static_cast<T>(0);
        check_cublas(
            Blas<T>::nrm2(engine->cublas, parameters, typed<T>(engine->d_delta), &difference_norm),
            "reduce Newton coefficient difference"
        );
        check_cublas(
            Blas<T>::nrm2(engine->cublas, parameters, typed<T>(engine->d_candidate), &beta_norm),
            "reduce Newton coefficient norm"
        );
        check_cuda(
            rh_cuda::launch_copy(
                typed<T>(engine->d_trial_beta), typed<T>(engine->d_candidate), parameters, engine->stream
            ),
            "commit accepted Newton candidate to workspace"
        );
        outcome.objective = candidate_objective;
        outcome.iterations = iteration;
        if (static_cast<double>(difference_norm) <=
            config->tolerance * (1.0 + static_cast<double>(beta_norm))) {
            outcome.converged = true;
            return outcome;
        }
    }
    outcome.iterations = config->max_iter;
    outcome.converged = false;
    return outcome;
}

template <typename T>
RhCudaStatus update_typed(
    RhCudaEngine* engine,
    const RhCudaHostBatch* batch,
    const RhCudaUnpenalizedConfig* config,
    RhCudaDiagnostics* diagnostics
) {
    validate_config(config, engine);
    validate_batch(batch, engine);
    check_header(diagnostics, "diagnostics");
    if (engine->n_samples_seen > std::numeric_limits<int64_t>::max() - batch->n_rows ||
        engine->batch_count == std::numeric_limits<int64_t>::max()) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "state counters would overflow");
    }

    const double n_total = engine->weight_sum + batch->batch_weight;
    const double bandwidth = bandwidth_for(engine, batch, config);
    copy_host_batch<T>(engine, batch);
    const T* weights = batch->sample_weight == nullptr ? nullptr : typed<T>(engine->d_weights);
    const SolveOutcome outcome = solve_unpenalized<T>(
        engine,
        static_cast<int>(batch->n_rows),
        weights,
        config,
        n_total,
        static_cast<T>(config->tau),
        static_cast<T>(bandwidth)
    );
    final_information<T>(
        engine,
        static_cast<int>(batch->n_rows),
        weights,
        static_cast<T>(config->tau),
        static_cast<T>(bandwidth)
    );

    /*
     * All state writes above target staging buffers.  Wait before swapping
     * pointers so a hard CUDA failure leaves the active state untouched.
     */
    check_cuda(cudaStreamSynchronize(engine->stream), "complete renewable CUDA update");
    std::swap(engine->d_coefficients, engine->d_trial_beta);
    std::swap(engine->d_information, engine->d_information_next);

    engine->n_samples_seen += batch->n_rows;
    engine->batch_count += 1;
    engine->previous_lambda = 0.0;
    engine->weight_sum = n_total;

    diagnostics->iterations = outcome.iterations;
    diagnostics->converged = outcome.converged ? 1 : 0;
    diagnostics->used_regularized_fallback = outcome.used_fallback ? 1 : 0;
    diagnostics->objective = outcome.objective;
    diagnostics->lambda_value = 0.0;
    diagnostics->bandwidth = bandwidth;
    return RH_CUDA_STATUS_SUCCESS;
}

template <typename T>
RhCudaStatus predict_typed(RhCudaEngine* engine, const RhCudaHostPrediction* request) {
    check_header(request, "host prediction");
    if (request->x_design == nullptr || request->prediction == nullptr || request->n_rows <= 0 ||
        request->n_rows > std::numeric_limits<int>::max() || request->n_columns != engine->n_parameters) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "host prediction has an invalid contiguous design matrix");
    }
    ensure_batch_capacity<T>(engine, request->n_rows);
    const size_t matrix = checked_elements(request->n_rows, engine->n_parameters, "prediction design");
    check_cuda(
        cudaMemcpyAsync(
            engine->d_design,
            request->x_design,
            matrix * sizeof(T),
            cudaMemcpyHostToDevice,
            engine->stream
        ),
        "copy prediction design to device"
    );
    const int rows = static_cast<int>(request->n_rows);
    const int parameters = static_cast<int>(engine->n_parameters);
    const T one = static_cast<T>(1);
    const T zero = static_cast<T>(0);
    check_cublas(
        Blas<T>::gemv(
            engine->cublas,
            CUBLAS_OP_T,
            parameters,
            rows,
            &one,
            typed<T>(engine->d_design),
            parameters,
            typed<T>(engine->d_coefficients),
            &zero,
            typed<T>(engine->d_residual)
        ),
        "compute host prediction"
    );
    check_cuda(
        cudaMemcpyAsync(
            request->prediction,
            engine->d_residual,
            static_cast<size_t>(rows) * sizeof(T),
            cudaMemcpyDeviceToHost,
            engine->stream
        ),
        "copy prediction to host"
    );
    check_cuda(cudaStreamSynchronize(engine->stream), "complete host prediction");
    return RH_CUDA_STATUS_SUCCESS;
}

template <typename Function>
RhCudaStatus guarded(RhCudaEngine* engine, Function&& function) noexcept {
    try {
        clear_error(g_last_error);
        set_device(engine);
        clear_error(engine->last_error);
        return function();
    } catch (const Failure& error) {
        set_error(g_last_error, error.what());
        if (engine != nullptr) {
            set_error(engine->last_error, error.what());
        }
        return error.status();
    } catch (const std::exception& error) {
        set_error(g_last_error, error.what());
        if (engine != nullptr) {
            set_error(engine->last_error, error.what());
        }
        return RH_CUDA_STATUS_INTERNAL_ERROR;
    } catch (...) {
        set_error(g_last_error, "unknown CUDA engine failure");
        if (engine != nullptr) {
            set_error(engine->last_error, "unknown CUDA engine failure");
        }
        return RH_CUDA_STATUS_INTERNAL_ERROR;
    }
}

template <typename Function>
RhCudaStatus global_guarded(Function&& function) noexcept {
    try {
        clear_error(g_last_error);
        return function();
    } catch (const Failure& error) {
        set_error(g_last_error, error.what());
        return error.status();
    } catch (const std::exception& error) {
        set_error(g_last_error, error.what());
        return RH_CUDA_STATUS_INTERNAL_ERROR;
    } catch (...) {
        set_error(g_last_error, "unknown CUDA runtime failure");
        return RH_CUDA_STATUS_INTERNAL_ERROR;
    }
}

}  // namespace

extern "C" {

uint32_t rh_cuda_abi_version(void) {
    return RH_CUDA_ABI_VERSION;
}

const char* rh_cuda_last_error(void) {
    return g_last_error.data();
}

RhCudaStatus rh_cuda_is_available(int32_t* available) {
    return global_guarded([&]() -> RhCudaStatus {
        if (available == nullptr) {
            fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "available output pointer must not be null");
        }
        *available = 0;
        int count = 0;
        const cudaError_t status = cudaGetDeviceCount(&count);
        if (status == cudaSuccess) {
            *available = count > 0 ? 1 : 0;
            return RH_CUDA_STATUS_SUCCESS;
        }
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver) {
            cudaGetLastError();
            return RH_CUDA_STATUS_SUCCESS;
        }
        check_cuda(status, "cudaGetDeviceCount");
        return RH_CUDA_STATUS_SUCCESS;
    });
}

RhCudaStatus rh_cuda_device_count(int32_t* count) {
    return global_guarded([&]() -> RhCudaStatus {
        if (count == nullptr) {
            fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "count output pointer must not be null");
        }
        *count = 0;
        int devices = 0;
        const cudaError_t status = cudaGetDeviceCount(&devices);
        if (status == cudaSuccess) {
            *count = devices;
            return RH_CUDA_STATUS_SUCCESS;
        }
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver) {
            cudaGetLastError();
            return RH_CUDA_STATUS_SUCCESS;
        }
        check_cuda(status, "cudaGetDeviceCount");
        return RH_CUDA_STATUS_SUCCESS;
    });
}

RhCudaStatus rh_cuda_runtime_info(RhCudaRuntimeInfo* info) {
    try {
        clear_error(g_last_error);
        check_header(info, "runtime info");
        int runtime = 0;
        int driver = 0;
        int devices = 0;
        check_cuda(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
        check_cuda(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
        const cudaError_t device_status = cudaGetDeviceCount(&devices);
        if (device_status != cudaSuccess && device_status != cudaErrorNoDevice &&
            device_status != cudaErrorInsufficientDriver) {
            check_cuda(device_status, "cudaGetDeviceCount");
        }
        if (device_status != cudaSuccess) {
            cudaGetLastError();
            devices = 0;
        }
        info->runtime_version = runtime;
        info->driver_version = driver;
        info->device_count = devices;
        return RH_CUDA_STATUS_SUCCESS;
    } catch (const Failure& error) {
        set_error(g_last_error, error.what());
        return error.status();
    } catch (const std::exception& error) {
        set_error(g_last_error, error.what());
        return RH_CUDA_STATUS_INTERNAL_ERROR;
    } catch (...) {
        set_error(g_last_error, "unknown CUDA runtime-info failure");
        return RH_CUDA_STATUS_INTERNAL_ERROR;
    }
}

RhCudaStatus rh_cuda_engine_create(
    const RhCudaEngineOptions* options,
    RhCudaEngine** out_engine
) {
    clear_error(g_last_error);
    if (out_engine == nullptr) {
        set_error(g_last_error, "out_engine must not be null");
        return RH_CUDA_STATUS_INVALID_ARGUMENT;
    }
    *out_engine = nullptr;
    RhCudaEngine* engine = nullptr;
    try {
        validate_options(options);
        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (options->device_id >= device_count) {
            fail(RH_CUDA_STATUS_UNAVAILABLE, "requested CUDA device is unavailable");
        }
        check_cuda(cudaSetDevice(options->device_id), "cudaSetDevice");
        engine = new RhCudaEngine();
        engine->dtype = options->dtype;
        engine->device_id = options->device_id;
        engine->n_parameters = options->n_parameters;
        check_cuda(cudaStreamCreateWithFlags(&engine->stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        check_cublas(cublasCreate(&engine->cublas), "cublasCreate");
        check_cublas(cublasSetStream(engine->cublas, engine->stream), "cublasSetStream");
        check_cublas(cublasSetPointerMode(engine->cublas, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
        check_cublas(cublasSetMathMode(engine->cublas, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");
        check_cusolver(cusolverDnCreate(&engine->solver), "cusolverDnCreate");
        check_cusolver(cusolverDnSetStream(engine->solver, engine->stream), "cusolverDnSetStream");
        if (engine->dtype == RH_CUDA_DTYPE_FLOAT32) {
            allocate_static_buffers<float>(engine);
        } else {
            allocate_static_buffers<double>(engine);
        }
        *out_engine = engine;
        return RH_CUDA_STATUS_SUCCESS;
    } catch (const Failure& error) {
        set_error(g_last_error, error.what());
        delete engine;
        return error.status();
    } catch (const std::exception& error) {
        set_error(g_last_error, error.what());
        delete engine;
        return RH_CUDA_STATUS_INTERNAL_ERROR;
    } catch (...) {
        set_error(g_last_error, "unknown CUDA engine creation failure");
        delete engine;
        return RH_CUDA_STATUS_INTERNAL_ERROR;
    }
}

RhCudaStatus rh_cuda_engine_destroy(RhCudaEngine* engine) {
    delete engine;
    return RH_CUDA_STATUS_SUCCESS;
}

RhCudaStatus rh_cuda_engine_restore(RhCudaEngine* engine, const RhCudaHostStateView* state) {
    return guarded(engine, [&]() -> RhCudaStatus {
        check_header(state, "host state");
        if (state->coefficients == nullptr || state->information == nullptr || state->n_samples_seen < 0 ||
            state->batch_count < 0 || !finite_nonnegative(state->previous_lambda) ||
            !finite_nonnegative(state->weight_sum) ||
            (state->n_samples_seen > 0 && state->weight_sum == 0.0)) {
            fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "host state violates renewable invariants");
        }
        const size_t parameters = static_cast<size_t>(engine->n_parameters);
        const size_t square = checked_elements(engine->n_parameters, engine->n_parameters, "restored information");
        if (engine->dtype == RH_CUDA_DTYPE_FLOAT32) {
            validate_finite_state(
                  static_cast<const float*>(state->coefficients),
                  static_cast<const float*>(state->information),
                  parameters
            );
        } else {
            validate_finite_state(
                  static_cast<const double*>(state->coefficients),
                  static_cast<const double*>(state->information),
                  parameters
            );
        }
        if (engine->dtype == RH_CUDA_DTYPE_FLOAT32) {
            check_cuda(
                cudaMemcpyAsync(
                    engine->d_coefficients, state->coefficients, parameters * sizeof(float), cudaMemcpyHostToDevice,
                    engine->stream
                ),
                "restore float32 coefficients"
            );
            check_cuda(
                cudaMemcpyAsync(
                    engine->d_information_next, state->information, square * sizeof(float), cudaMemcpyHostToDevice,
                    engine->stream
                ),
                "restore float32 information"
            );
            check_cuda(
                rh_cuda::launch_transpose(
                    typed<float>(engine->d_information_next),
                    typed<float>(engine->d_information),
                    engine->n_parameters,
                    engine->n_parameters,
                    engine->stream
                ),
                "transpose restored float32 information"
            );
        } else {
            check_cuda(
                cudaMemcpyAsync(
                    engine->d_coefficients, state->coefficients, parameters * sizeof(double), cudaMemcpyHostToDevice,
                    engine->stream
                ),
                "restore float64 coefficients"
            );
            check_cuda(
                cudaMemcpyAsync(
                    engine->d_information_next, state->information, square * sizeof(double), cudaMemcpyHostToDevice,
                    engine->stream
                ),
                "restore float64 information"
            );
            check_cuda(
                rh_cuda::launch_transpose(
                    typed<double>(engine->d_information_next),
                    typed<double>(engine->d_information),
                    engine->n_parameters,
                    engine->n_parameters,
                    engine->stream
                ),
                "transpose restored float64 information"
            );
        }
        check_cuda(cudaStreamSynchronize(engine->stream), "complete state restore");
        engine->n_samples_seen = state->n_samples_seen;
        engine->batch_count = state->batch_count;
        engine->previous_lambda = state->previous_lambda;
        engine->weight_sum = state->weight_sum;
        return RH_CUDA_STATUS_SUCCESS;
    });
}

RhCudaStatus rh_cuda_engine_copy_state(RhCudaEngine* engine, RhCudaHostState* state) {
    return guarded(engine, [&]() -> RhCudaStatus {
        check_header(state, "host state");
        if (state->coefficients == nullptr || state->information == nullptr) {
            fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "host state output buffers must not be null");
        }
        const size_t parameters = static_cast<size_t>(engine->n_parameters);
        const size_t square = checked_elements(engine->n_parameters, engine->n_parameters, "copied information");
        const size_t item_size = engine->dtype == RH_CUDA_DTYPE_FLOAT32 ? sizeof(float) : sizeof(double);
        check_cuda(
            cudaMemcpyAsync(
                state->coefficients,
                engine->d_coefficients,
                parameters * item_size,
                cudaMemcpyDeviceToHost,
                engine->stream
            ),
            "copy coefficients to host"
        );
        if (engine->dtype == RH_CUDA_DTYPE_FLOAT32) {
            check_cuda(
                rh_cuda::launch_transpose(
                    typed<float>(engine->d_information),
                    typed<float>(engine->d_information_next),
                    engine->n_parameters,
                    engine->n_parameters,
                    engine->stream
                ),
                "transpose float32 information for host"
            );
        } else {
            check_cuda(
                rh_cuda::launch_transpose(
                    typed<double>(engine->d_information),
                    typed<double>(engine->d_information_next),
                    engine->n_parameters,
                    engine->n_parameters,
                    engine->stream
                ),
                "transpose float64 information for host"
            );
        }
        check_cuda(
            cudaMemcpyAsync(
                state->information,
                engine->d_information_next,
                square * item_size,
                cudaMemcpyDeviceToHost,
                engine->stream
            ),
            "copy information to host"
        );
        check_cuda(cudaStreamSynchronize(engine->stream), "complete state copy");
        state->n_samples_seen = engine->n_samples_seen;
        state->batch_count = engine->batch_count;
        state->previous_lambda = engine->previous_lambda;
        state->weight_sum = engine->weight_sum;
        return RH_CUDA_STATUS_SUCCESS;
    });
}

RhCudaStatus rh_cuda_engine_update_host(
    RhCudaEngine* engine,
    const RhCudaHostBatch* batch,
    const RhCudaUnpenalizedConfig* config,
    RhCudaDiagnostics* diagnostics
) {
    return guarded(engine, [&]() -> RhCudaStatus {
        if (engine->dtype == RH_CUDA_DTYPE_FLOAT32) {
            return update_typed<float>(engine, batch, config, diagnostics);
        }
        return update_typed<double>(engine, batch, config, diagnostics);
    });
}

RhCudaStatus rh_cuda_engine_predict_host(RhCudaEngine* engine, const RhCudaHostPrediction* request) {
    return guarded(engine, [&]() -> RhCudaStatus {
        if (engine->dtype == RH_CUDA_DTYPE_FLOAT32) {
            return predict_typed<float>(engine, request);
        }
        return predict_typed<double>(engine, request);
    });
}

RhCudaStatus rh_cuda_engine_synchronize(RhCudaEngine* engine) {
    return guarded(engine, [&]() -> RhCudaStatus {
        check_cuda(cudaStreamSynchronize(engine->stream), "engine synchronize");
        return RH_CUDA_STATUS_SUCCESS;
    });
}

const char* rh_cuda_engine_last_error(const RhCudaEngine* engine) {
    if (engine == nullptr) {
        return "engine is null";
    }
    return engine->last_error.data();
}

}  // extern "C"
