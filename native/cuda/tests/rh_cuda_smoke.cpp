#include "rh_cuda.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace {

template <typename T>
void initialize(T* value) {
    value->abi_version = RH_CUDA_ABI_VERSION;
    value->struct_size = sizeof(T);
}

bool close(double actual, double expected, double tolerance = 1e-5) {
    return std::abs(actual - expected) <= tolerance;
}

bool check(RhCudaStatus status, const char* operation) {
    if (status == RH_CUDA_STATUS_SUCCESS) {
        return true;
    }
    std::cerr << operation << " failed: " << rh_cuda_last_error() << '\n';
    return false;
}

}  // namespace

int main() {
    if (rh_cuda_is_available(nullptr) != RH_CUDA_STATUS_INVALID_ARGUMENT ||
        std::strlen(rh_cuda_last_error()) == 0) {
        std::cerr << "availability null-pointer error did not stay inside the C ABI\n";
        return 1;
    }
    if (rh_cuda_device_count(nullptr) != RH_CUDA_STATUS_INVALID_ARGUMENT ||
        std::strlen(rh_cuda_last_error()) == 0) {
        std::cerr << "device-count null-pointer error did not stay inside the C ABI\n";
        return 1;
    }

    int32_t available = 0;
    if (!check(rh_cuda_is_available(&available), "rh_cuda_is_available")) {
        return 1;
    }
    if (!available) {
        std::cout << "SKIP: no CUDA device available\n";
        return 0;
    }

    RhCudaEngineOptions options{};
    initialize(&options);
    options.dtype = RH_CUDA_DTYPE_FLOAT64;
    options.device_id = 0;
    options.n_parameters = 2;

    RhCudaEngine* engine = nullptr;
    if (!check(rh_cuda_engine_create(&options, &engine), "rh_cuda_engine_create")) {
        return 1;
    }
    RhCudaEngineFeatures features{};
    initialize(&features);
    if (!check(rh_cuda_engine_features(engine, &features), "rh_cuda_engine_features strict") ||
        features.requested_flags != 0 || features.enabled_flags != 0) {
        std::cerr << "strict engine unexpectedly enabled CUDA tuning\n";
        rh_cuda_engine_destroy(engine);
        return 1;
    }
    rh_cuda_engine_destroy(engine);

    options.reserved0 = RH_CUDA_ENGINE_FLAG_FAST_MATH;
    if (rh_cuda_engine_create(&options, &engine) != RH_CUDA_STATUS_INVALID_ARGUMENT) {
        std::cerr << "float64 engine accepted fast precision\n";
        if (engine != nullptr) {
            rh_cuda_engine_destroy(engine);
        }
        return 1;
    }

    options.reserved0 = RH_CUDA_ENGINE_FLAG_CUDA_GRAPHS;
    if (!check(rh_cuda_engine_create(&options, &engine), "rh_cuda_engine_create graph")) {
        return 1;
    }

    double coefficients[2] = {0.0, 0.0};
    double asymmetric_information[4] = {1.0, 2.0, 3.0, 4.0};
    RhCudaHostStateView state{};
    initialize(&state);
    state.coefficients = coefficients;
    state.information = asymmetric_information;
    state.n_samples_seen = 1;
    state.batch_count = 1;
    state.previous_lambda = 0.0;
    state.weight_sum = 1.0;
    if (!check(rh_cuda_engine_restore(engine, &state), "rh_cuda_engine_restore asymmetric")) {
        rh_cuda_engine_destroy(engine);
        return 1;
    }

    double copied_coefficients[2] = {};
    double copied_information[4] = {};
    RhCudaHostState copied{};
    initialize(&copied);
    copied.coefficients = copied_coefficients;
    copied.information = copied_information;
    if (!check(rh_cuda_engine_copy_state(engine, &copied), "rh_cuda_engine_copy_state asymmetric")) {
        rh_cuda_engine_destroy(engine);
        return 1;
    }
    if (std::memcmp(asymmetric_information, copied_information, sizeof(asymmetric_information)) != 0) {
        std::cerr << "row-major information round trip changed an asymmetric matrix\n";
        rh_cuda_engine_destroy(engine);
        return 1;
    }

    double zero_information[4] = {};
    state.information = zero_information;
    state.n_samples_seen = 0;
    state.batch_count = 0;
    state.weight_sum = 0.0;
    if (!check(rh_cuda_engine_restore(engine, &state), "rh_cuda_engine_restore zero")) {
        rh_cuda_engine_destroy(engine);
        return 1;
    }

    /* Keep the initial residuals in the Huber quadratic region. */
    const double design[8] = {-1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0};
    const double target[4] = {-0.1, 0.1, 0.3, 0.5};
    const double weights[4] = {1.0, 1.0, 1.0, 1.0};
    RhCudaHostBatch batch{};
    initialize(&batch);
    batch.x_design = design;
    batch.y = target;
    batch.sample_weight = weights;
    batch.n_rows = 4;
    batch.n_columns = 2;
    batch.batch_weight = 4.0;

    RhCudaUnpenalizedConfig config{};
    initialize(&config);
    config.n_features_in = 1;
    config.max_iter = 100;
    config.tau = 1.345;
    config.bandwidth_scale = 1.0;
    config.tolerance = 1e-8;
    config.ridge = 1e-8;

    RhCudaDiagnostics diagnostics{};
    initialize(&diagnostics);
    double fused_coefficients[2] = {};
    double fused_information[4] = {};
    RhCudaHostState fused{};
    initialize(&fused);
    fused.coefficients = fused_coefficients;
    fused.information = fused_information;
    if (!check(
            rh_cuda_engine_update_host_with_state(engine, &batch, &config, &diagnostics, &fused),
            "rh_cuda_engine_update_host_with_state"
        )) {
        rh_cuda_engine_destroy(engine);
        return 1;
    }
    if (!check(rh_cuda_engine_copy_state(engine, &copied), "rh_cuda_engine_copy_state update")) {
        rh_cuda_engine_destroy(engine);
        return 1;
    }
    if (std::memcmp(fused_coefficients, copied_coefficients, sizeof(fused_coefficients)) != 0 ||
        std::memcmp(fused_information, copied_information, sizeof(fused_information)) != 0 ||
        fused.n_samples_seen != copied.n_samples_seen || fused.batch_count != copied.batch_count ||
        fused.previous_lambda != copied.previous_lambda || fused.weight_sum != copied.weight_sum) {
        std::cerr << "fused update state differs from a subsequent state copy\n";
        rh_cuda_engine_destroy(engine);
        return 1;
    }
    if (!close(copied_coefficients[0], 0.2, 1e-4) || !close(copied_coefficients[1], 0.1, 1e-4)) {
        std::cerr << "unexpected fitted coefficients: " << copied_coefficients[0] << ", "
                  << copied_coefficients[1] << '\n';
        rh_cuda_engine_destroy(engine);
        return 1;
    }
    if (copied.n_samples_seen != 4 || copied.batch_count != 1 || !close(copied.weight_sum, 4.0)) {
        std::cerr << "unexpected renewable counters after update\n";
        rh_cuda_engine_destroy(engine);
        return 1;
    }

    double predictions[4] = {};
    RhCudaHostPrediction prediction{};
    initialize(&prediction);
    prediction.x_design = design;
    prediction.prediction = predictions;
    prediction.n_rows = 4;
    prediction.n_columns = 2;
    if (!check(rh_cuda_engine_predict_host(engine, &prediction), "rh_cuda_engine_predict_host")) {
        rh_cuda_engine_destroy(engine);
        return 1;
    }
    for (int index = 0; index < 4; ++index) {
        if (!close(predictions[index], target[index], 1e-4)) {
            std::cerr << "unexpected prediction at " << index << ": " << predictions[index] << '\n';
            rh_cuda_engine_destroy(engine);
            return 1;
        }
    }

    /*
     * Portable checkpoints may contain a general information matrix.  The
     * frozen Python update applies every entry in its Newton gradient, so the
     * dense factorization must not silently mirror one triangle as Cholesky
     * would.  With X=I in the quadratic Huber region, the first Newton step is
     * exactly (information + I)^-1 y.
     */
    double general_information[4] = {4.0, 0.3, 0.1, 3.0};
    const double identity_design[4] = {1.0, 0.0, 0.0, 1.0};
    const double general_target[2] = {0.2, -0.1};
    const double general_weights[2] = {1.0, 1.0};
    coefficients[0] = 0.0;
    coefficients[1] = 0.0;
    state.coefficients = coefficients;
    state.information = general_information;
    state.n_samples_seen = 1;
    state.batch_count = 1;
    state.weight_sum = 1.0;
    if (!check(rh_cuda_engine_restore(engine, &state), "restore general information")) {
        rh_cuda_engine_destroy(engine);
        return 1;
    }
    batch.x_design = identity_design;
    batch.y = general_target;
    batch.sample_weight = general_weights;
    batch.n_rows = 2;
    batch.n_columns = 2;
    batch.batch_weight = 2.0;
    config.n_features_in = 1;
    config.ridge = 0.0;
    initialize(&diagnostics);
    if (!check(rh_cuda_engine_update_host(engine, &batch, &config, &diagnostics), "general-information update")) {
        rh_cuda_engine_destroy(engine);
        return 1;
    }
    if (!check(rh_cuda_engine_copy_state(engine, &copied), "copy general-information state")) {
        rh_cuda_engine_destroy(engine);
        return 1;
    }
    const double determinant = 5.0 * 4.0 - 0.3 * 0.1;
    const double expected_general_coefficients[2] = {
        (4.0 * 0.2 - 0.3 * -0.1) / determinant,
        (-0.1 * 0.2 + 5.0 * -0.1) / determinant,
    };
    if (diagnostics.used_regularized_fallback ||
        !close(copied_coefficients[0], expected_general_coefficients[0], 1e-10) ||
        !close(copied_coefficients[1], expected_general_coefficients[1], 1e-10) ||
        !close(copied_information[0], 5.0, 1e-10) ||
        !close(copied_information[1], 0.3, 1e-10) ||
        !close(copied_information[2], 0.1, 1e-10) ||
        !close(copied_information[3], 4.0, 1e-10) || copied.n_samples_seen != 3 ||
        copied.batch_count != 2 || !close(copied.weight_sum, 3.0, 1e-10)) {
        std::cerr << "general information matrix did not use the full pivoted-LU solve\n";
        rh_cuda_engine_destroy(engine);
        return 1;
    }

    /* Singular pivoted-LU factorization must take the minimum-norm SVD fallback. */
    state.coefficients = coefficients;
    state.information = zero_information;
    state.n_samples_seen = 0;
    state.batch_count = 0;
    state.weight_sum = 0.0;
    coefficients[0] = 0.0;
    coefficients[1] = 0.0;
    if (!check(rh_cuda_engine_restore(engine, &state), "rh_cuda_engine_restore rank deficient")) {
        rh_cuda_engine_destroy(engine);
        return 1;
    }
    const double singular_design[8] = {-1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0};
    const double singular_target[4] = {-0.2, 0.0, 0.2, 0.4};
    const double singular_weights[4] = {1.0, 2.0, 1.0, 3.0};
    batch.x_design = singular_design;
    batch.y = singular_target;
    batch.sample_weight = singular_weights;
    batch.batch_weight = 7.0;
    config.n_features_in = 2;
    config.ridge = 0.0;
    initialize(&diagnostics);
    if (!check(rh_cuda_engine_update_host(engine, &batch, &config, &diagnostics), "rank-deficient update")) {
        rh_cuda_engine_destroy(engine);
        return 1;
    }
    if (!diagnostics.used_regularized_fallback) {
        std::cerr << "rank-deficient update did not use the minimum-norm fallback\n";
        rh_cuda_engine_destroy(engine);
        return 1;
    }
    if (!check(rh_cuda_engine_copy_state(engine, &copied), "copy rank-deficient state")) {
        rh_cuda_engine_destroy(engine);
        return 1;
    }
    if (!close(copied_coefficients[0], 0.1, 1e-4) || !close(copied_coefficients[1], 0.1, 1e-4)) {
        std::cerr << "unexpected minimum-norm coefficients: " << copied_coefficients[0] << ", "
                  << copied_coefficients[1] << '\n';
        rh_cuda_engine_destroy(engine);
        return 1;
    }

    initialize(&features);
    if (!check(rh_cuda_engine_features(engine, &features), "rh_cuda_engine_features graph") ||
        features.requested_flags != RH_CUDA_ENGINE_FLAG_CUDA_GRAPHS ||
        features.graph_captures + features.graph_fallbacks == 0) {
        std::cerr << "CUDA Graph request was neither captured nor safely disabled\n";
        rh_cuda_engine_destroy(engine);
        return 1;
    }

    rh_cuda_engine_destroy(engine);
    std::cout << "rh_cuda_smoke passed\n";
    return 0;
}
