#include "batch.cuh"
#include "blas_traits.cuh"
#include "engine_internal.cuh"
#include "huber_kernels.cuh"
#include "linear_solver.cuh"
#include "objective.cuh"
#include "pipeline.cuh"
#include "workspace.cuh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace rh_cuda::engine {

struct SolveOutcome {
    int64_t iterations = 0;
    bool converged = false;
    double objective = 0.0;
    bool used_fallback = false;
    bool residual_is_current = true;
};

template <typename T>
SolveOutcome solve_unpenalized(
    RhCudaEngine* engine,
    int rows,
    const T* weights,
    const RhCudaUpdateConfig* config,
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
    ).objective;
    CandidateObjectiveGraph candidate_graph(engine);
    ObjectiveTerms<T> candidate_terms;
    candidate_terms.previous_beta = typed<T>(engine->d_trial_beta);

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
        bool cholesky_status_pending = solve_direction<T>(
            engine,
            config->ridge > 0.0,
            &outcome.used_fallback
        );

        bool accepted = false;
        outcome.residual_is_current = false;
        double candidate_objective = outcome.objective;
        ObjectiveResult candidate_result;
        int backtrack = 0;
        while (backtrack <= 26) {
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
            candidate_result = candidate_graph.evaluate<T>(
                rows,
                typed<T>(engine->d_candidate),
                weights,
                tau,
                n_total,
                candidate_terms
            );
            candidate_objective = candidate_result.objective;
            if (cholesky_status_pending) {
                cholesky_status_pending = false;
                if (!cholesky_candidate_is_valid<T>(engine, &outcome.used_fallback)) {
                    // POTRF failed. LU/SVD has replaced the tentative
                    // direction; discard this candidate and restart the line
                    // search from a full step without advancing the counter.
                    backtrack = 0;
                    continue;
                }
            }
            if (candidate_objective <= outcome.objective) {
                accepted = true;
                break;
            }
            ++backtrack;
        }
        if (!accepted) {
            outcome.iterations = iteration;
            outcome.converged = false;
            return outcome;
        }

        check_cuda(
            rh_cuda::launch_copy(
                typed<T>(engine->d_trial_beta), typed<T>(engine->d_candidate), parameters, engine->stream
            ),
            "commit accepted Newton candidate to workspace"
        );
        outcome.residual_is_current = true;
        outcome.objective = candidate_objective;
        outcome.iterations = iteration;
        if (candidate_result.difference_norm <=
            config->tolerance * (1.0 + candidate_result.beta_norm)) {
            outcome.converged = true;
            return outcome;
        }
    }
    outcome.iterations = config->max_iter;
    outcome.converged = false;
    return outcome;
}

/*
 * LAMM proximal-gradient transition for the L1 penalty, mirroring the NumPy
 * reference and the Rust CPU engine step for step: majorize with a quadratic
 * of curvature phi, soft-threshold every coordinate except the trailing
 * intercept, double phi until the candidate's smooth objective sits under
 * the majorizer, then halve phi (floored at 1e-8) after an accepted step.
 *
 * The residual resident in d_residual always belongs to d_trial_beta at the
 * top of an iteration: the initial objective evaluates it, and every accepted
 * candidate objective re-establishes it for the next gradient.
 */
template <typename T>
SolveOutcome solve_l1(
    RhCudaEngine* engine,
    int rows,
    const T* weights,
    const RhCudaUpdateConfig* config,
    double n_total,
    T tau,
    double lambda_value,
    int64_t penalized_count,
    const T* penalty_sign,
    double penalty_scale
) {
    const int parameters = static_cast<int>(engine->n_parameters);
    check_cuda(
        rh_cuda::launch_copy(
            typed<T>(engine->d_trial_beta), typed<T>(engine->d_coefficients), parameters, engine->stream
        ),
        "initialize proximal coefficients"
    );

    ObjectiveTerms<T> base_terms;
    base_terms.penalty_sign = penalty_sign;
    base_terms.penalty_scale = penalty_scale;
    SolveOutcome outcome;
    outcome.objective = smooth_objective<T>(
        engine, rows, typed<T>(engine->d_trial_beta), weights, tau, n_total, base_terms
    ).objective;

    CandidateObjectiveGraph candidate_graph(engine);
    ObjectiveTerms<T> candidate_terms = base_terms;
    candidate_terms.previous_beta = typed<T>(engine->d_trial_beta);
    candidate_terms.gradient = typed<T>(engine->d_gradient);
    double phi = 1.0;

    for (int iteration = 1; iteration <= static_cast<int>(config->max_iter); ++iteration) {
        compute_l1_gradient<T>(
            engine,
            rows,
            typed<T>(engine->d_trial_beta),
            weights,
            tau,
            n_total,
            penalty_sign,
            penalty_scale
        );

        bool accepted = false;
        outcome.residual_is_current = false;
        ObjectiveResult candidate_result;
        for (int attempt = 0; attempt < 40; ++attempt) {
            check_cuda(
                rh_cuda::launch_soft_threshold_candidate(
                    typed<T>(engine->d_trial_beta),
                    typed<T>(engine->d_gradient),
                    static_cast<T>(1.0 / phi),
                    static_cast<T>(lambda_value / phi),
                    typed<T>(engine->d_candidate),
                    parameters,
                    penalized_count,
                    engine->stream
                ),
                "form proximal candidate"
            );
            candidate_result = candidate_graph.evaluate<T>(
                rows,
                typed<T>(engine->d_candidate),
                weights,
                tau,
                n_total,
                candidate_terms
            );
            const double upper_bound = outcome.objective + candidate_result.gradient_dot +
                0.5 * phi * candidate_result.difference_norm * candidate_result.difference_norm;
            if (candidate_result.objective <= upper_bound + 1.0e-12) {
                accepted = true;
                break;
            }
            phi *= 2.0;
        }
        if (!accepted) {
            outcome.iterations = iteration;
            outcome.converged = false;
            return outcome;
        }

        check_cuda(
            rh_cuda::launch_copy(
                typed<T>(engine->d_trial_beta), typed<T>(engine->d_candidate), parameters, engine->stream
            ),
            "commit accepted proximal candidate to workspace"
        );
        outcome.residual_is_current = true;
        outcome.objective = candidate_result.objective;
        outcome.iterations = iteration;
        phi = std::max(phi * 0.5, 1.0e-8);
        if (candidate_result.difference_norm <=
            config->tolerance * (1.0 + candidate_result.beta_norm)) {
            outcome.converged = true;
            return outcome;
        }
    }
    outcome.iterations = config->max_iter;
    outcome.converged = false;
    return outcome;
}

template <typename T>
void enqueue_state_copy(
    RhCudaEngine* engine,
    const T* coefficients,
    const T* information,
    RhCudaHostState* state
) {
    check_header(state, "host state");
    if (state->coefficients == nullptr || state->information == nullptr) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "host state output buffers must not be null");
    }
    const size_t parameters = static_cast<size_t>(engine->n_parameters);
    const size_t square = checked_elements(
        engine->n_parameters, engine->n_parameters, "copied information"
    );
    check_cuda(
        cudaMemcpyAsync(
            state->coefficients,
            coefficients,
            parameters * sizeof(T),
            cudaMemcpyDeviceToHost,
            engine->stream
        ),
        "copy coefficients to host"
    );

    const T* portable_information = information;
    if (!engine->information_is_symmetric) {
        // Internal matrices are column-major while checkpoints are row-major.
        // A symmetric matrix has the same byte layout in both conventions;
        // only a restored general matrix requires an explicit transpose.
        check_cuda(
            rh_cuda::launch_transpose(
                information,
                typed<T>(engine->d_gram),
                engine->n_parameters,
                engine->n_parameters,
                engine->stream
            ),
            "transpose information for host"
        );
        portable_information = typed<T>(engine->d_gram);
    }
    check_cuda(
        cudaMemcpyAsync(
            state->information,
            portable_information,
            square * sizeof(T),
            cudaMemcpyDeviceToHost,
            engine->stream
        ),
        "copy information to host"
    );
}

void fill_state_metadata(
    const RhCudaEngine* engine,
    RhCudaHostState* state
) {
    state->n_samples_seen = engine->n_samples_seen;
    state->batch_count = engine->batch_count;
    state->previous_lambda = engine->previous_lambda;
    state->weight_sum = engine->weight_sum;
}

template <typename T>
RhCudaStatus update_typed(
    RhCudaEngine* engine,
    const BatchView& batch,
    const RhCudaUpdateConfig* config,
    RhCudaDiagnostics* diagnostics,
    RhCudaHostState* exported_state
) {
    validate_config(config, engine);
    validate_batch(batch, config, engine);
    check_header(diagnostics, "diagnostics");
    if (exported_state != nullptr) {
        check_header(exported_state, "host state");
        if (exported_state->coefficients == nullptr || exported_state->information == nullptr) {
            fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "host state output buffers must not be null");
        }
    }
    if (engine->n_samples_seen > std::numeric_limits<int64_t>::max() - batch.n_rows ||
        engine->batch_count == std::numeric_limits<int64_t>::max()) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "state counters would overflow");
    }

    const bool l1 = config->penalty == RH_CUDA_PENALTY_L1;
    const double n_total = engine->weight_sum + batch.batch_weight;
    const double bandwidth = bandwidth_for(engine, batch.batch_weight, config);
    const double lambda_value = lambda_for(engine, batch.batch_weight, config);
    // The trailing coordinate is the intercept exactly when the engine carries
    // one parameter more than the caller has features; it is never penalized.
    const int64_t penalized_count = engine->n_parameters -
        (engine->n_parameters == config->n_features_in + 1 ? 1 : 0);
    copy_batch<T>(engine, batch);
    // DLPack keeps the producer target alive until this call returns. Alias it
    // directly during the solver instead of copying it into owned workspace;
    // the guard restores the allocation pointer on every exception path.
    ScopedPointerAlias target_alias(
        &engine->d_y,
        batch.y,
        batch.copy_kind == cudaMemcpyDeviceToDevice
    );
    const T* weights = batch.sample_weight == nullptr ? nullptr : typed<T>(engine->d_weights);
    SolveOutcome outcome;
    if (l1) {
        // The reference applies the historical subgradient only once a
        // previous batch exists; before that the state carries no lambda.
        const T* penalty_sign = nullptr;
        double penalty_scale = 0.0;
        if (engine->n_samples_seen > 0) {
            check_cuda(
                rh_cuda::launch_penalty_sign(
                    typed<T>(engine->d_coefficients),
                    typed<T>(engine->d_penalty_sign),
                    engine->n_parameters,
                    penalized_count,
                    engine->stream
                ),
                "form L1 historical subgradient"
            );
            penalty_sign = typed<T>(engine->d_penalty_sign);
            penalty_scale = engine->weight_sum / n_total * engine->previous_lambda;
        }
        outcome = solve_l1<T>(
            engine,
            static_cast<int>(batch.n_rows),
            weights,
            config,
            n_total,
            static_cast<T>(config->tau),
            lambda_value,
            penalized_count,
            penalty_sign,
            penalty_scale
        );
    } else {
        outcome = solve_unpenalized<T>(
            engine,
            static_cast<int>(batch.n_rows),
            weights,
            config,
            n_total,
            static_cast<T>(config->tau),
            static_cast<T>(bandwidth)
        );
    }
    final_information<T>(
        engine,
        static_cast<int>(batch.n_rows),
        weights,
        static_cast<T>(config->tau),
        static_cast<T>(bandwidth),
        outcome.residual_is_current
    );
    if (l1) {
        // The reported objective adds lambda * ||beta||_1 over the penalized
        // coordinates. Queue the reduction behind the solve so it shares the
        // update's single transactional synchronization.
        check_cublas(
            Blas<T>::asum(
                engine->cublas_reduction,
                static_cast<int>(penalized_count),
                typed<T>(engine->d_trial_beta),
                typed<T>(engine->d_reduction_results) + 6
            ),
            "reduce final L1 norm"
        );
        check_cuda(
            cudaMemcpyAsync(
                typed<T>(engine->h_reduction_results) + 6,
                typed<T>(engine->d_reduction_results) + 6,
                sizeof(T),
                cudaMemcpyDeviceToHost,
                engine->stream
            ),
            "read final L1 norm"
        );
    }

    if (exported_state != nullptr) {
        // Both outputs still live in staging buffers. Queue their D2H copies
        // before the update's transactional completion so callers pay for one
        // stream wait rather than update + copy_state waits.
        enqueue_state_copy<T>(
            engine,
            typed<T>(engine->d_trial_beta),
            typed<T>(engine->d_information_next),
            exported_state
        );
    }

    /*
     * All state writes above target staging buffers.  Wait before swapping
     * pointers so a hard CUDA failure leaves the active state untouched.
     */
    check_cuda(cudaStreamSynchronize(engine->stream), "complete renewable CUDA update");
    double objective = outcome.objective;
    if (l1) {
        objective += lambda_value *
            static_cast<double>(typed<T>(engine->h_reduction_results)[6]);
    }
    std::swap(engine->d_coefficients, engine->d_trial_beta);
    std::swap(engine->d_information, engine->d_information_next);

    engine->n_samples_seen += batch.n_rows;
    engine->batch_count += 1;
    engine->previous_lambda = lambda_value;
    engine->weight_sum = n_total;
    if (exported_state != nullptr) {
        fill_state_metadata(engine, exported_state);
    }

    diagnostics->iterations = outcome.iterations;
    diagnostics->converged = outcome.converged ? 1 : 0;
    diagnostics->used_regularized_fallback = outcome.used_fallback ? 1 : 0;
    diagnostics->objective = objective;
    diagnostics->lambda_value = lambda_value;
    diagnostics->bandwidth = bandwidth;
    return RH_CUDA_STATUS_SUCCESS;
}

template <typename T>
RhCudaStatus predict_typed(RhCudaEngine* engine, const RhCudaPrediction* request) {
    check_header(request, "prediction");
    if (request->x_design == nullptr || request->prediction == nullptr || request->n_rows <= 0 ||
        request->n_rows > std::numeric_limits<int>::max() || request->reserved0 != 0) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "prediction request is malformed");
    }
    if (request->input_location != RH_CUDA_MEMORY_HOST &&
        request->input_location != RH_CUDA_MEMORY_DEVICE) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "prediction input_location is unknown");
    }
    validate_intercept_layout(request->n_features_in, engine);
    if (request->n_columns != engine->n_parameters && request->n_columns != request->n_features_in) {
        fail(
            RH_CUDA_STATUS_INVALID_ARGUMENT,
            "prediction columns must match n_features_in or the expanded engine parameters"
        );
    }
    const bool device_input = request->input_location == RH_CUDA_MEMORY_DEVICE;
    if (device_input) {
        validate_device_pointer(engine, request->x_design, "inspect device prediction X");
    }
    ensure_batch_capacity<T>(engine, request->n_rows);
    const size_t matrix = checked_elements(request->n_rows, request->n_columns, "prediction design");
    const cudaMemcpyKind copy_kind = device_input ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;

    const T* design = typed<T>(engine->d_design);
    if (request->n_columns == engine->n_parameters) {
        if (device_input) {
            // An expanded device design is read in place: the GEMV below is
            // the only consumer, so a staging copy would buy nothing.
            design = typed<T>(request->x_design);
        } else {
            check_cuda(
                cudaMemcpyAsync(
                    engine->d_design,
                    request->x_design,
                    matrix * sizeof(T),
                    copy_kind,
                    engine->stream
                ),
                "copy prediction design to device"
            );
        }
    } else {
        const T* features = typed<T>(request->x_design);
        if (!device_input) {
            check_cuda(
                cudaMemcpyAsync(
                    engine->d_weighted_design,
                    request->x_design,
                    matrix * sizeof(T),
                    copy_kind,
                    engine->stream
                ),
                "copy prediction features to device"
            );
            features = typed<T>(engine->d_weighted_design);
        }
        check_cuda(
            rh_cuda::launch_append_intercept(
                features,
                typed<T>(engine->d_design),
                request->n_rows,
                request->n_columns,
                engine->stream
            ),
            "append prediction intercept column on device"
        );
    }
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
            design,
            parameters,
            typed<T>(engine->d_coefficients),
            &zero,
            typed<T>(engine->d_residual)
        ),
        "compute prediction"
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
    check_cuda(cudaStreamSynchronize(engine->stream), "complete prediction");
    return RH_CUDA_STATUS_SUCCESS;
}

// Explicit instantiation: the engine is only ever float or double, and a
// missing pair fails the link instead of silently duplicating a definition.
template void enqueue_state_copy<float>(
    RhCudaEngine*, const float*, const float*, RhCudaHostState*
);
template void enqueue_state_copy<double>(
    RhCudaEngine*, const double*, const double*, RhCudaHostState*
);
template RhCudaStatus update_typed<float>(
    RhCudaEngine*, const BatchView&, const RhCudaUpdateConfig*,
    RhCudaDiagnostics*, RhCudaHostState*
);
template RhCudaStatus update_typed<double>(
    RhCudaEngine*, const BatchView&, const RhCudaUpdateConfig*,
    RhCudaDiagnostics*, RhCudaHostState*
);
template RhCudaStatus predict_typed<float>(RhCudaEngine*, const RhCudaPrediction*);
template RhCudaStatus predict_typed<double>(RhCudaEngine*, const RhCudaPrediction*);

}  // namespace rh_cuda::engine
