#ifndef RENEWABLE_HUBER_RH_CUDA_BLAS_TRAITS_CUH
#define RENEWABLE_HUBER_RH_CUDA_BLAS_TRAITS_CUH

/*
 * float/double dispatch for the cuBLAS and cuSOLVER entry points the engine
 * uses. Header-only: these are class-template full specializations whose
 * members are all defined in-class and therefore implicitly inline.
 *
 * Kept out of engine_state.cuh on purpose -- every consumer of the engine type
 * would otherwise be forced to pull in <cublas_v2.h> and <cusolverDn.h>.
 */

#include "engine_internal.cuh"

#include <cublas_v2.h>
#include <cusolverDn.h>

namespace rh_cuda::engine {

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
    static cublasStatus_t syrkx(
        cublasHandle_t h,
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
        return cublasSsyrkx(
            h,
            CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc
        );
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
    static cublasStatus_t syrkx(
        cublasHandle_t h,
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
        return cublasDsyrkx(
            h,
            CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc
        );
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
    static cusolverStatus_t potrf_buffer_size(
        cusolverDnHandle_t h, int n, float* a, int lda, int* lwork
    ) {
        return cusolverDnSpotrf_bufferSize(h, CUBLAS_FILL_MODE_LOWER, n, a, lda, lwork);
    }
    static cusolverStatus_t potrf(
        cusolverDnHandle_t h, int n, float* a, int lda, float* work, int lwork, int* info
    ) {
        return cusolverDnSpotrf(
            h, CUBLAS_FILL_MODE_LOWER, n, a, lda, work, lwork, info
        );
    }
    static cusolverStatus_t potrs(
        cusolverDnHandle_t h, int n, const float* a, int lda, float* rhs, int ldb, int* info
    ) {
        return cusolverDnSpotrs(
            h, CUBLAS_FILL_MODE_LOWER, n, 1, a, lda, rhs, ldb, info
        );
    }
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
    static cusolverStatus_t potrf_buffer_size(
        cusolverDnHandle_t h, int n, double* a, int lda, int* lwork
    ) {
        return cusolverDnDpotrf_bufferSize(h, CUBLAS_FILL_MODE_LOWER, n, a, lda, lwork);
    }
    static cusolverStatus_t potrf(
        cusolverDnHandle_t h, int n, double* a, int lda, double* work, int lwork, int* info
    ) {
        return cusolverDnDpotrf(
            h, CUBLAS_FILL_MODE_LOWER, n, a, lda, work, lwork, info
        );
    }
    static cusolverStatus_t potrs(
        cusolverDnHandle_t h, int n, const double* a, int lda, double* rhs, int ldb, int* info
    ) {
        return cusolverDnDpotrs(
            h, CUBLAS_FILL_MODE_LOWER, n, 1, a, lda, rhs, ldb, info
        );
    }
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

}  // namespace rh_cuda::engine

#endif  // RENEWABLE_HUBER_RH_CUDA_BLAS_TRAITS_CUH
