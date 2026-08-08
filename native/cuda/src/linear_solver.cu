#include "blas_traits.cuh"
#include "engine_internal.cuh"
#include "huber_kernels.cuh"
#include "linear_solver.cuh"
#include "workspace.cuh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace rh_cuda::engine {

template <typename T>
void solve_minimum_norm_svd(RhCudaEngine* engine, bool* used_fallback) {
    ensure_svd_workspace<T>(engine);
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
void solve_direction_lu(RhCudaEngine* engine, bool* used_fallback) {
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
            typed<T>(engine->d_factor_work),
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
        cudaMemcpyAsync(
            &info,
            engine->d_solver_info,
            sizeof(int),
            cudaMemcpyDeviceToHost,
            engine->stream
        ),
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
bool solve_direction(
    RhCudaEngine* engine,
    bool allow_cholesky,
    bool* used_fallback
) {
    if (!allow_cholesky || !engine->information_is_symmetric) {
        solve_direction_lu<T>(engine, used_fallback);
        return false;
    }
    const int parameters = static_cast<int>(engine->n_parameters);
    const size_t square = checked_elements(
        engine->n_parameters, engine->n_parameters, "Cholesky factor"
    );
    check_cuda(
        cudaMemcpyAsync(
            engine->d_factor,
            engine->d_hessian,
            square * sizeof(T),
            cudaMemcpyDeviceToDevice,
            engine->stream
        ),
        "copy Hessian for Cholesky"
    );
    check_cusolver(
        Solver<T>::potrf(
            engine->solver,
            parameters,
            typed<T>(engine->d_factor),
            parameters,
            typed<T>(engine->d_factor_work),
            engine->factor_lwork,
            engine->d_solver_info
        ),
        "cusolverDn*potrf"
    );
    check_cublas(
        Blas<T>::copy(
            engine->cublas,
            parameters,
            typed<T>(engine->d_gradient),
            typed<T>(engine->d_direction)
        ),
        "copy gradient into Cholesky solve RHS"
    );
    check_cusolver(
        Solver<T>::potrs(
            engine->solver,
            parameters,
            typed<T>(engine->d_factor),
            parameters,
            typed<T>(engine->d_direction),
            parameters,
            engine->d_solver_info + 1
        ),
        "cusolverDn*potrs"
    );
    check_cuda(
        cudaMemcpyAsync(
            engine->h_solver_info,
            engine->d_solver_info,
            2 * sizeof(int),
            cudaMemcpyDeviceToHost,
            engine->stream
        ),
        "queue Cholesky factorization and solve info"
    );
    // The immediately following candidate objective synchronizes this same
    // stream. Deferring the host inspection removes one round trip per Newton
    // iteration while the pinned destination remains engine-owned and alive.
    return true;
}

template <typename T>
bool cholesky_candidate_is_valid(RhCudaEngine* engine, bool* used_fallback) {
    const int* info = engine->h_solver_info;
    if (info[0] < 0) {
        fail(
            RH_CUDA_STATUS_INTERNAL_ERROR,
            "Cholesky factorization received an invalid cuSOLVER argument"
        );
    }
    if (info[0] > 0) {
        /*
         * The renewable Hessian is symmetric positive semidefinite in exact
         * arithmetic, but numerical rank deficiency can still make it
         * singular. POTRS is deliberately queued before reading POTRF's
         * device status, then its tentative output is discarded here and the
         * established LU -> SVD correctness path recomputes the direction.
         * This removes a host round trip from the positive-definite path
         * without weakening the fallback semantics.
         */
        solve_direction_lu<T>(engine, used_fallback);
        return false;
    }
    if (info[1] != 0) {
        fail(
            RH_CUDA_STATUS_NUMERICAL_ERROR,
            info[1] < 0 ? "Cholesky solve received an invalid cuSOLVER argument"
                        : "Cholesky solve failed to produce a finite direction"
        );
    }
    return true;
}

// Explicit instantiation: the engine is only ever float or double, and a
// missing pair fails the link instead of silently duplicating a definition.
template bool solve_direction<float>(RhCudaEngine*, bool, bool*);
template bool solve_direction<double>(RhCudaEngine*, bool, bool*);
template bool cholesky_candidate_is_valid<float>(RhCudaEngine*, bool*);
template bool cholesky_candidate_is_valid<double>(RhCudaEngine*, bool*);

}  // namespace rh_cuda::engine
