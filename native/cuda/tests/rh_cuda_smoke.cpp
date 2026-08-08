#include "rh_cuda.h"

#include <cuda_runtime_api.h>

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

/*
 * The cases below are written against the two helpers that follow rather than
 * main()'s inline style, because each of them needs two or three engines and
 * repeating the destroy-and-return teardown that many times is how this file
 * would grow past 500 lines.  main()'s original sequence is left untouched: it
 * is the regression baseline for the engine.cu split.
 */

struct EngineHandle {
    RhCudaEngine* value = nullptr;

    ~EngineHandle() {
        if (value != nullptr) {
            rh_cuda_engine_destroy(value);
        }
    }

    EngineHandle() = default;
    EngineHandle(const EngineHandle&) = delete;
    EngineHandle& operator=(const EngineHandle&) = delete;
};

#define REQUIRE(condition, message)                                                  \
    do {                                                                             \
        if (!(condition)) {                                                          \
            std::cerr << __FILE__ << ':' << __LINE__ << ": " << (message) << " ["    \
                      << #condition << "]\n";                                        \
            return false;                                                            \
        }                                                                            \
    } while (0)

/* Device allocation that frees itself however the case exits. */
struct DeviceBuffer {
    void* value = nullptr;

    ~DeviceBuffer() {
        if (value != nullptr) {
            cudaFree(value);
        }
    }

    bool upload(const double* host, size_t count) {
        if (cudaMalloc(&value, count * sizeof(double)) != cudaSuccess) {
            return false;
        }
        return cudaMemcpy(value, host, count * sizeof(double), cudaMemcpyHostToDevice) ==
               cudaSuccess;
    }

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

bool create_float64_engine(EngineHandle& handle, int64_t n_parameters) {
    RhCudaEngineOptions options{};
    initialize(&options);
    options.dtype = RH_CUDA_DTYPE_FLOAT64;
    options.device_id = 0;
    options.n_parameters = n_parameters;
    return rh_cuda_engine_create(&options, &handle.value) == RH_CUDA_STATUS_SUCCESS;
}

RhCudaUnpenalizedConfig unpenalized_config(int64_t n_features_in) {
    RhCudaUnpenalizedConfig config{};
    initialize(&config);
    config.n_features_in = n_features_in;
    config.max_iter = 100;
    config.tau = 1.345;
    config.bandwidth_scale = 1.0;
    config.tolerance = 1e-8;
    config.ridge = 1e-8;
    return config;
}

/* Room for the widest shape any case below uses. */
constexpr int kMaxParameters = 4;

struct Fit {
    double coefficients[kMaxParameters]{};
    double information[kMaxParameters * kMaxParameters]{};
    RhCudaHostState state{};
    RhCudaDiagnostics diagnostics{};
};

bool restore_zero_state(RhCudaEngine* engine, int64_t n_parameters) {
    double coefficients[kMaxParameters] = {};
    double information[kMaxParameters * kMaxParameters] = {};
    RhCudaHostStateView state{};
    initialize(&state);
    state.coefficients = coefficients;
    state.information = information;
    (void)n_parameters;
    return rh_cuda_engine_restore(engine, &state) == RH_CUDA_STATUS_SUCCESS;
}

/* One update on a fresh engine restored to the canonical empty state. */
bool fit_host_batch(
    int64_t n_parameters,
    int64_t n_features_in,
    const double* x_design,
    int64_t n_rows,
    int64_t n_columns,
    const double* y,
    Fit& fit
) {
    EngineHandle handle;
    if (!create_float64_engine(handle, n_parameters) ||
        !restore_zero_state(handle.value, n_parameters)) {
        return false;
    }

    RhCudaHostBatch batch{};
    initialize(&batch);
    batch.x_design = x_design;
    batch.y = y;
    batch.sample_weight = nullptr;
    batch.n_rows = n_rows;
    batch.n_columns = n_columns;
    batch.batch_weight = static_cast<double>(n_rows);

    const RhCudaUnpenalizedConfig config = unpenalized_config(n_features_in);
    initialize(&fit.state);
    initialize(&fit.diagnostics);
    fit.state.coefficients = fit.coefficients;
    fit.state.information = fit.information;
    return rh_cuda_engine_update_host_with_state(
               handle.value, &batch, &config, &fit.diagnostics, &fit.state
           ) == RH_CUDA_STATUS_SUCCESS;
}

bool fits_agree(const Fit& wide, const Fit& narrow, int64_t n_parameters) {
    /*
     * The two paths differ only in how d_design gets filled -- a straight
     * memcpy versus launch_append_intercept.  Every cuBLAS call downstream then
     * sees identical shapes and identical values, so the results should be
     * bit-identical; 1e-12 is slack, not necessity.  Comparing them directly
     * rather than against hand-derived numbers catches an intercept in the
     * wrong column, a wrong row stride, a missing element, and a wrong fill
     * value all at once.
     */
    for (int64_t index = 0; index < n_parameters; ++index) {
        if (!close(narrow.coefficients[index], wide.coefficients[index], 1e-12)) {
            std::cerr << "coefficient " << index << " differs: wide " << wide.coefficients[index]
                      << " vs narrow " << narrow.coefficients[index] << '\n';
            return false;
        }
    }
    for (int64_t index = 0; index < n_parameters * n_parameters; ++index) {
        if (!close(narrow.information[index], wide.information[index], 1e-12)) {
            std::cerr << "information " << index << " differs: wide " << wide.information[index]
                      << " vs narrow " << narrow.information[index] << '\n';
            return false;
        }
    }
    return narrow.state.n_samples_seen == wide.state.n_samples_seen &&
           narrow.state.batch_count == wide.state.batch_count &&
           close(narrow.state.weight_sum, wide.state.weight_sum, 1e-12);
}

/* The same four rows main() fits, with and without the intercept column. */
const double kWideDesign2[8] = {-1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0};
const double kNarrowDesign2[4] = {-1.0, 0.0, 1.0, 2.0};
const double kTarget[4] = {-0.1, 0.1, 0.3, 0.5};

bool case_intercept_append_host_p2() {
    Fit wide;
    Fit narrow;
    REQUIRE(fit_host_batch(2, 1, kWideDesign2, 4, 2, kTarget, wide), "wide host fit failed");
    REQUIRE(fit_host_batch(2, 1, kNarrowDesign2, 4, 1, kTarget, narrow), "narrow host fit failed");
    REQUIRE(fits_agree(wide, narrow, 2), "device-appended intercept changed the fit");
    return true;
}

bool case_intercept_append_host_p3() {
    /*
     * p=2 cannot distinguish a kernel indexing features[column * rows + row]
     * from one indexing features[row * feature_columns + column]: with a single
     * feature column the two are the same expression.  Two features is the
     * smallest shape that actually pins the row-major layout.
     */
    const double wide_design[12] = {
        -1.0, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, -1.0, 1.0,
    };
    const double narrow_design[8] = {-1.0, 2.0, 0.0, 1.0, 1.0, 0.0, 2.0, -1.0};

    Fit wide;
    Fit narrow;
    REQUIRE(fit_host_batch(3, 2, wide_design, 4, 3, kTarget, wide), "wide p=3 host fit failed");
    REQUIRE(
        fit_host_batch(3, 2, narrow_design, 4, 2, kTarget, narrow), "narrow p=3 host fit failed"
    );
    REQUIRE(fits_agree(wide, narrow, 3), "device-appended intercept mis-indexed a 2-feature batch");
    return true;
}

/* One update from a device-resident batch, mirroring the DLPack contract. */
bool fit_device_batch(
    int64_t n_parameters,
    int64_t n_features_in,
    const double* x_design,
    int64_t n_rows,
    int64_t n_columns,
    const double* y,
    Fit& fit
) {
    EngineHandle handle;
    if (!create_float64_engine(handle, n_parameters) ||
        !restore_zero_state(handle.value, n_parameters)) {
        return false;
    }

    DeviceBuffer x;
    DeviceBuffer target;
    if (!x.upload(x_design, static_cast<size_t>(n_rows * n_columns)) ||
        !target.upload(y, static_cast<size_t>(n_rows))) {
        return false;
    }
    /*
     * Blocking copies on the default stream, deliberately: using the engine
     * stream asynchronously here would turn this into a stream-ordering test
     * rather than a test of the device-input path.
     */
    if (cudaDeviceSynchronize() != cudaSuccess) {
        return false;
    }

    RhCudaDeviceBatch batch{};
    initialize(&batch);
    batch.x_design = x.value;
    batch.y = target.value;
    batch.sample_weight = nullptr;
    batch.n_rows = n_rows;
    batch.n_columns = n_columns;
    batch.batch_weight = static_cast<double>(n_rows);

    const RhCudaUnpenalizedConfig config = unpenalized_config(n_features_in);
    initialize(&fit.state);
    initialize(&fit.diagnostics);
    fit.state.coefficients = fit.coefficients;
    fit.state.information = fit.information;
    return rh_cuda_engine_update_device_with_state(
               handle.value, &batch, &config, &fit.diagnostics, &fit.state
           ) == RH_CUDA_STATUS_SUCCESS;
}

bool case_intercept_append_device() {
    /*
     * The highest-value new case: on the device path copy_batch skips the
     * staging copy entirely and launch_append_intercept reads the caller's
     * pointer directly.  Nothing else exercises that branch.
     */
    Fit host;
    Fit device;
    REQUIRE(fit_host_batch(2, 1, kWideDesign2, 4, 2, kTarget, host), "host reference fit failed");
    REQUIRE(
        fit_device_batch(2, 1, kNarrowDesign2, 4, 1, kTarget, device), "narrow device fit failed"
    );
    REQUIRE(fits_agree(host, device, 2), "device-input intercept append changed the fit");
    return true;
}

bool case_update_device_then_host() {
    /*
     * update_typed aliases engine->d_y to the caller's device pointer for the
     * duration of the call.  If that guard ever stops restoring the original,
     * the next host update writes into memory the producer already freed.  A
     * subsequent host update that still succeeds and still agrees with the
     * reference is a direct, cheap regression test for it.
     */
    EngineHandle handle;
    REQUIRE(create_float64_engine(handle, 2), "engine creation failed");
    REQUIRE(restore_zero_state(handle.value, 2), "zero-state restore failed");

    DeviceBuffer x;
    DeviceBuffer target;
    REQUIRE(x.upload(kWideDesign2, 8), "device upload of X failed");
    REQUIRE(target.upload(kTarget, 4), "device upload of y failed");
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess, "device upload did not complete");

    RhCudaDeviceBatch device_batch{};
    initialize(&device_batch);
    device_batch.x_design = x.value;
    device_batch.y = target.value;
    device_batch.n_rows = 4;
    device_batch.n_columns = 2;
    device_batch.batch_weight = 4.0;

    const RhCudaUnpenalizedConfig config = unpenalized_config(1);
    Fit device;
    initialize(&device.state);
    initialize(&device.diagnostics);
    device.state.coefficients = device.coefficients;
    device.state.information = device.information;
    REQUIRE(
        rh_cuda_engine_update_device_with_state(
            handle.value, &device_batch, &config, &device.diagnostics, &device.state
        ) == RH_CUDA_STATUS_SUCCESS,
        "wide device update failed"
    );

    RhCudaHostBatch host_batch{};
    initialize(&host_batch);
    host_batch.x_design = kWideDesign2;
    host_batch.y = kTarget;
    host_batch.n_rows = 4;
    host_batch.n_columns = 2;
    host_batch.batch_weight = 4.0;

    Fit host;
    initialize(&host.state);
    initialize(&host.diagnostics);
    host.state.coefficients = host.coefficients;
    host.state.information = host.information;
    REQUIRE(
        rh_cuda_engine_update_host_with_state(
            handle.value, &host_batch, &config, &host.diagnostics, &host.state
        ) == RH_CUDA_STATUS_SUCCESS,
        "host update after a device update failed; d_y may not have been restored"
    );
    REQUIRE(host.state.n_samples_seen == 8, "the second update did not accumulate");
    REQUIRE(host.state.batch_count == 2, "the second update did not count");
    return true;
}

bool case_device_batch_rejects_host_pointer() {
    EngineHandle handle;
    REQUIRE(create_float64_engine(handle, 2), "engine creation failed");
    REQUIRE(restore_zero_state(handle.value, 2), "zero-state restore failed");

    RhCudaDeviceBatch batch{};
    initialize(&batch);
    batch.x_design = kWideDesign2;  /* host memory, deliberately */
    batch.y = kTarget;
    batch.n_rows = 4;
    batch.n_columns = 2;
    batch.batch_weight = 4.0;

    const RhCudaUnpenalizedConfig config = unpenalized_config(1);
    RhCudaDiagnostics diagnostics{};
    initialize(&diagnostics);
    double coefficients[2] = {};
    double information[4] = {};
    RhCudaHostState state{};
    initialize(&state);
    state.coefficients = coefficients;
    state.information = information;

    const RhCudaStatus status = rh_cuda_engine_update_device_with_state(
        handle.value, &batch, &config, &diagnostics, &state
    );
    /*
     * The exact code is deliberately not pinned: cudaPointerGetAttributes
     * reports an unregistered host pointer as cudaErrorInvalidValue on older
     * runtimes (-> CUDA_ERROR) and as success with type Unregistered on CUDA 11
     * and later (-> INVALID_ARGUMENT).  Both are correct rejections.
     */
    REQUIRE(status != RH_CUDA_STATUS_SUCCESS, "a host pointer was accepted as device memory");
    REQUIRE(
        std::strlen(rh_cuda_engine_last_error(handle.value)) != 0,
        "rejection left no error message"
    );
    return true;
}

bool case_status_survives_translation_units() {
    /*
     * Regression test for the one way this refactor can fail silently. After
     * the split validate_batch lives in batch.cu and the catch that maps
     * Failure to a status code lives in the C API layer. If the exception type
     * were ever duplicated per translation unit, the catch would stop matching
     * and every error would degrade to INTERNAL_ERROR while the numerics
     * stayed correct.
     */
    EngineHandle handle;
    REQUIRE(create_float64_engine(handle, 2), "engine creation failed");
    REQUIRE(restore_zero_state(handle.value, 2), "zero-state restore failed");

    RhCudaHostBatch batch{};
    initialize(&batch);
    batch.x_design = kWideDesign2;
    batch.y = kTarget;
    batch.n_rows = 4;
    batch.n_columns = 5;  /* matches neither n_parameters nor n_features_in */
    batch.batch_weight = 4.0;

    const RhCudaUnpenalizedConfig config = unpenalized_config(1);
    RhCudaDiagnostics diagnostics{};
    initialize(&diagnostics);
    const RhCudaStatus status =
        rh_cuda_engine_update_host(handle.value, &batch, &config, &diagnostics);
    REQUIRE(
        status == RH_CUDA_STATUS_INVALID_ARGUMENT,
        "a Failure thrown outside the C API layer lost its status crossing translation units"
    );
    return true;
}

bool case_intercept_invariant_is_enforced() {
    /*
     * n_parameters must be n_features_in or n_features_in + 1.  A wider gap
     * would let the device-side append fill only part of d_design and leave
     * the rest stale, so the C ABI has to reject it rather than solve against
     * uninitialized columns.
     */
    EngineHandle handle;
    REQUIRE(create_float64_engine(handle, 4), "engine creation failed");
    REQUIRE(restore_zero_state(handle.value, 4), "zero-state restore failed");

    RhCudaHostBatch batch{};
    initialize(&batch);
    batch.x_design = kNarrowDesign2;
    batch.y = kTarget;
    batch.n_rows = 4;
    batch.n_columns = 1;
    batch.batch_weight = 4.0;

    const RhCudaUnpenalizedConfig config = unpenalized_config(1);
    RhCudaDiagnostics diagnostics{};
    initialize(&diagnostics);
    REQUIRE(
        rh_cuda_engine_update_host(handle.value, &batch, &config, &diagnostics) ==
            RH_CUDA_STATUS_INVALID_ARGUMENT,
        "n_features_in = 1 with n_parameters = 4 was accepted"
    );
    return true;
}

bool case_linked_abi_version_matches_header() {
    REQUIRE(
        rh_cuda_abi_version() == RH_CUDA_ABI_VERSION,
        "the linked library reports a different ABI version than this header"
    );
    return true;
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

    /*
     * Cases added alongside the engine.cu split.  They run after the original
     * sequence above, which stays untouched as the regression baseline.
     */
    const struct {
        const char* name;
        bool (*run)();
    } cases[] = {
        {"linked_abi_version_matches_header", case_linked_abi_version_matches_header},
        {"intercept_append_host_p2", case_intercept_append_host_p2},
        {"intercept_append_host_p3", case_intercept_append_host_p3},
        {"intercept_append_device", case_intercept_append_device},
        {"update_device_then_host", case_update_device_then_host},
        {"device_batch_rejects_host_pointer", case_device_batch_rejects_host_pointer},
        {"status_survives_translation_units", case_status_survives_translation_units},
        {"intercept_invariant_is_enforced", case_intercept_invariant_is_enforced},
    };
    for (const auto& entry : cases) {
        if (!entry.run()) {
            std::cerr << "case " << entry.name << " failed\n";
            return 1;
        }
    }

    std::cout << "rh_cuda_smoke passed\n";
    return 0;
}
