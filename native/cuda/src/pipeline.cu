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
    /// Residual of d_trial_beta, left by the last accepted round; null when
    /// the solve stopped without accepting, so it must be recomputed.
    const void* residual = nullptr;
};

/// The Newton line search tries steps 2^0 .. 2^-kMaxBacktracks.
constexpr int kMaxBacktracks = 26;
/// The LAMM search tries at most this many curvatures phi, 2 phi, ...
constexpr int kMaxProximalAttempts = 40;

/*
 * Round widths.  Speculating is not free: a round of four reads X once but
 * does four candidates' dot products, which shows on float64, where the GPU is
 * compute-bound.  A Newton search therefore opens with the full step alone
 * unless it is expected to backtrack -- the first iteration from an empty
 * state, whose step overshoots far, or any iteration after one that
 * backtracked.  LAMM halves phi after every accepted step and often doubles it
 * straight back, so it opens with two curvatures.  Every later round is full
 * width.  The width never changes a value, only how many rounds it takes.
 */
constexpr int kNewtonOpeningWidth = 1;
constexpr int kProximalOpeningWidth = 2;

/*
 * Damped Newton for penalty NONE.  Each line-search round evaluates the next
 * kCandidateRoundWidth step sizes together; the first accepted one in step
 * order wins, exactly as if they had been tried one at a time, and a round's
 * width never changes a candidate's value (see huber_kernels.cuh).
 */
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

    CandidateRoundEvaluator rounds(engine);
    const ObjectiveTerms<T> terms;
    ObjectiveResult evaluated[rh_cuda::kCandidateRoundWidth];
    SolveOutcome outcome;
    rounds.evaluate<T>(rows, weights, tau, n_total, terms, CandidateRound{}, evaluated);
    outcome.objective = evaluated[0].objective;
    const T* residual = rounds.residual<T>(rows, 0);
    bool expect_backtracking = engine->n_samples_seen == 0;

    for (int iteration = 1; iteration <= static_cast<int>(config->max_iter); ++iteration) {
        compute_gradient_hessian<T>(
            engine,
            rows,
            typed<T>(engine->d_trial_beta),
            residual,
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

        int accepted = -1;
        int backtrack = 0;
        while (backtrack <= kMaxBacktracks) {
            CandidateRound round;
            round.form = rh_cuda::kCandidateNewton;
            const int width = backtrack == 0 && !expect_backtracking
                ? kNewtonOpeningWidth
                : rh_cuda::kCandidateRoundWidth;
            round.width = std::min(width, kMaxBacktracks + 1 - backtrack);
            for (int k = 0; k < round.width; ++k) {
                round.scale[k] = std::ldexp(1.0, -(backtrack + k));
            }
            rounds.evaluate<T>(rows, weights, tau, n_total, terms, round, evaluated);
            if (cholesky_status_pending) {
                cholesky_status_pending = false;
                if (!cholesky_candidate_is_valid<T>(engine, &outcome.used_fallback)) {
                    // POTRF failed. LU/SVD has replaced the tentative
                    // direction; discard this round and restart the line
                    // search from a full step.
                    backtrack = 0;
                    continue;
                }
            }
            for (int k = 0; k < round.width; ++k) {
                if (evaluated[k].objective <= outcome.objective) {
                    accepted = k;
                    break;
                }
            }
            if (accepted >= 0) {
                break;
            }
            backtrack += round.width;
        }
        if (accepted < 0) {
            outcome.iterations = iteration;
            outcome.converged = false;
            outcome.residual = nullptr;
            return outcome;
        }

        check_cuda(
            rh_cuda::launch_copy(
                typed<T>(engine->d_trial_beta), rounds.candidate<T>(accepted), parameters, engine->stream
            ),
            "commit accepted Newton candidate to workspace"
        );
        residual = rounds.residual<T>(rows, accepted);
        outcome.residual = residual;
        outcome.objective = evaluated[accepted].objective;
        outcome.iterations = iteration;
        expect_backtracking = backtrack + accepted > 0;
        if (evaluated[accepted].difference_norm <=
            config->tolerance * (1.0 + evaluated[accepted].beta_norm)) {
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
 * A round evaluates the next kCandidateRoundWidth curvatures together and
 * accepts the first that satisfies the bound, as the sequential search would.
 *
 * The residual a round leaves for its accepted candidate always belongs to
 * d_trial_beta at the top of an iteration: the initial round evaluates it, and
 * every accepted candidate re-establishes it for the next gradient.
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

    // Every round of the update shares these terms, including the initial
    // one, whose copy form neither reads nor reduces the gradient.
    ObjectiveTerms<T> terms;
    terms.gradient = typed<T>(engine->d_gradient);
    terms.penalty_sign = penalty_sign;
    terms.penalty_scale = penalty_scale;
    terms.penalized_count = penalized_count;
    CandidateRoundEvaluator rounds(engine);
    ObjectiveResult evaluated[rh_cuda::kCandidateRoundWidth];
    SolveOutcome outcome;
    rounds.evaluate<T>(rows, weights, tau, n_total, terms, CandidateRound{}, evaluated);
    outcome.objective = evaluated[0].objective;
    const T* residual = rounds.residual<T>(rows, 0);
    double phi = 1.0;

    for (int iteration = 1; iteration <= static_cast<int>(config->max_iter); ++iteration) {
        compute_l1_gradient<T>(
            engine,
            rows,
            typed<T>(engine->d_trial_beta),
            residual,
            weights,
            tau,
            n_total,
            penalty_sign,
            penalty_scale
        );

        int accepted = -1;
        for (int attempt = 0; attempt < kMaxProximalAttempts && accepted < 0;) {
            CandidateRound round;
            round.form = rh_cuda::kCandidateProximal;
            const int width = attempt == 0 ? kProximalOpeningWidth : rh_cuda::kCandidateRoundWidth;
            round.width = std::min(width, kMaxProximalAttempts - attempt);
            double curvature[rh_cuda::kCandidateRoundWidth] = {};
            double next_phi = phi;
            for (int k = 0; k < round.width; ++k) {
                curvature[k] = next_phi;
                round.scale[k] = 1.0 / next_phi;
                round.threshold[k] = lambda_value / next_phi;
                next_phi *= 2.0;
            }
            rounds.evaluate<T>(rows, weights, tau, n_total, terms, round, evaluated);
            for (int k = 0; k < round.width; ++k) {
                const double upper_bound = outcome.objective + evaluated[k].gradient_dot +
                    0.5 * curvature[k] * evaluated[k].difference_norm *
                        evaluated[k].difference_norm;
                if (evaluated[k].objective <= upper_bound + 1.0e-12) {
                    accepted = k;
                    phi = curvature[k];
                    break;
                }
            }
            if (accepted < 0) {
                phi = next_phi;
            }
            attempt += round.width;
        }
        if (accepted < 0) {
            outcome.iterations = iteration;
            outcome.converged = false;
            outcome.residual = nullptr;
            return outcome;
        }

        check_cuda(
            rh_cuda::launch_copy(
                typed<T>(engine->d_trial_beta), rounds.candidate<T>(accepted), parameters, engine->stream
            ),
            "commit accepted proximal candidate to workspace"
        );
        residual = rounds.residual<T>(rows, accepted);
        outcome.residual = residual;
        outcome.objective = evaluated[accepted].objective;
        outcome.iterations = iteration;
        phi = std::max(phi * 0.5, 1.0e-8);
        if (evaluated[accepted].difference_norm <=
            config->tolerance * (1.0 + evaluated[accepted].beta_norm)) {
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
        static_cast<const T*>(outcome.residual),
        weights,
        static_cast<T>(config->tau),
        static_cast<T>(bandwidth)
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
