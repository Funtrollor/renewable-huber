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

RhCudaUpdateConfig unpenalized_config(int64_t n_features_in) {
    RhCudaUpdateConfig config{};
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

    const RhCudaUpdateConfig config = unpenalized_config(n_features_in);
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

    const RhCudaUpdateConfig config = unpenalized_config(n_features_in);
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

    const RhCudaUpdateConfig config = unpenalized_config(1);
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

    const RhCudaUpdateConfig config = unpenalized_config(1);
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

    const RhCudaUpdateConfig config = unpenalized_config(1);
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

    const RhCudaUpdateConfig config = unpenalized_config(1);
    RhCudaDiagnostics diagnostics{};
    initialize(&diagnostics);
    REQUIRE(
        rh_cuda_engine_update_host(handle.value, &batch, &config, &diagnostics) ==
            RH_CUDA_STATUS_INVALID_ARGUMENT,
        "n_features_in = 1 with n_parameters = 4 was accepted"
    );
    return true;
}

/*
 * ABI 2: penalty-aware update configuration and location-aware prediction.
 */

RhCudaUpdateConfig l1_config(int64_t n_features_in, double lambda_scale) {
    RhCudaUpdateConfig config = unpenalized_config(n_features_in);
    config.penalty = RH_CUDA_PENALTY_L1;
    config.lambda_scale = lambda_scale;
    config.max_iter = 400;
    config.tolerance = 1e-9;
    return config;
}

/* Two-feature stream shared with the NumPy/Rust reference values below. */
const double kL1Features1[12] = {
    -1.0, 0.5, 0.0, -1.0, 1.0, 0.25, 2.0, 1.5, -0.5, -0.5, 1.5, -1.25,
};
const double kL1Target1[6] = {-0.9, 0.35, 1.2, 2.6, -0.2, 1.1};
const double kL1Features2[8] = {0.5, 1.0, -1.5, 0.0, 1.0, -1.0, 0.25, 0.75};
const double kL1Target2[4] = {0.8, -1.4, 0.7, 0.6};
const double kL1Weights2[4] = {1.0, 2.0, 0.5, 0.0};

bool update_l1_host(
    RhCudaEngine* engine,
    const double* features,
    const double* target,
    const double* weights,
    int64_t rows,
    double batch_weight,
    const RhCudaUpdateConfig& config,
    Fit& fit
) {
    RhCudaHostBatch batch{};
    initialize(&batch);
    batch.x_design = features;
    batch.y = target;
    batch.sample_weight = weights;
    batch.n_rows = rows;
    batch.n_columns = 2;
    batch.batch_weight = batch_weight;
    initialize(&fit.state);
    initialize(&fit.diagnostics);
    fit.state.coefficients = fit.coefficients;
    fit.state.information = fit.information;
    return rh_cuda_engine_update_host_with_state(
               engine, &batch, &config, &fit.diagnostics, &fit.state
           ) == RH_CUDA_STATUS_SUCCESS;
}

bool case_l1_stream_matches_reference() {
    /*
     * Expected values come from the NumPy reference, which the Rust CPU engine
     * reproduces to ~1e-15 on this stream.  The solver deliberately runs to
     * max_iter; the GPU's reduction order moves the trajectory by far less
     * than the 1e-6 allowed here.  The second batch exercises the historical
     * subgradient, a zero weight, and non-unit weights.
     */
    EngineHandle handle;
    REQUIRE(create_float64_engine(handle, 3), "engine creation failed");
    REQUIRE(restore_zero_state(handle.value, 3), "zero-state restore failed");
    const RhCudaUpdateConfig config = l1_config(2, 0.8);

    Fit first;
    REQUIRE(
        update_l1_host(handle.value, kL1Features1, kL1Target1, nullptr, 6, 6.0, config, first),
        "first L1 update failed"
    );
    REQUIRE(close(first.coefficients[0], 0.6829540472953248, 1e-6), "first L1 coefficient 0");
    REQUIRE(first.coefficients[1] == 0.0, "first L1 update did not zero coefficient 1");
    REQUIRE(close(first.coefficients[2], 0.35018992533742943, 1e-6), "first L1 intercept");
    REQUIRE(close(first.diagnostics.objective, 0.3539251693367076, 1e-8), "first L1 objective");
    REQUIRE(
        close(first.diagnostics.lambda_value, 0.3657205604738795, 1e-12) &&
            close(first.state.previous_lambda, first.diagnostics.lambda_value, 0.0),
        "first L1 lambda was not committed as previous_lambda"
    );

    Fit second;
    REQUIRE(
        update_l1_host(handle.value, kL1Features2, kL1Target2, kL1Weights2, 4, 3.5, config, second),
        "second L1 update failed"
    );
    REQUIRE(close(second.coefficients[0], 0.8055087401784005, 1e-6), "second L1 coefficient 0");
    REQUIRE(second.coefficients[1] == 0.0, "second L1 update did not zero coefficient 1");
    REQUIRE(close(second.coefficients[2], 0.1792146584340598, 1e-6), "second L1 intercept");
    REQUIRE(close(second.diagnostics.objective, 0.23423173578534606, 1e-8), "second L1 objective");
    REQUIRE(close(second.state.previous_lambda, 0.29064522959496986, 1e-12), "second L1 lambda");
    REQUIRE(close(second.state.weight_sum, 9.5, 1e-12), "second L1 weight sum");
    REQUIRE(second.state.batch_count == 2 && second.state.n_samples_seen == 10, "L1 counters");

    const double probe[2] = {1.0, 2.0};
    double predicted[1] = {};
    RhCudaPrediction request{};
    initialize(&request);
    request.x_design = probe;
    request.prediction = predicted;
    request.n_rows = 1;
    request.n_columns = 2;
    request.n_features_in = 2;
    request.input_location = RH_CUDA_MEMORY_HOST;
    REQUIRE(rh_cuda_engine_predict(handle.value, &request) == RH_CUDA_STATUS_SUCCESS, "L1 predict");
    REQUIRE(close(predicted[0], 0.9847233986124603, 1e-6), "L1 narrow host prediction");
    return true;
}

bool case_l1_never_penalizes_the_intercept() {
    EngineHandle handle;
    REQUIRE(create_float64_engine(handle, 3), "engine creation failed");
    REQUIRE(restore_zero_state(handle.value, 3), "zero-state restore failed");
    Fit fit;
    REQUIRE(
        update_l1_host(
            handle.value, kL1Features1, kL1Target1, nullptr, 6, 6.0, l1_config(2, 50.0), fit
        ),
        "heavily penalized update failed"
    );
    REQUIRE(fit.coefficients[0] == 0.0 && fit.coefficients[1] == 0.0, "features were not zeroed");
    REQUIRE(close(fit.coefficients[2], 0.6125009324249108, 1e-6), "the intercept was penalized");
    return true;
}

bool case_l1_rejections_leave_state_untouched() {
    EngineHandle handle;
    REQUIRE(create_float64_engine(handle, 3), "engine creation failed");
    REQUIRE(restore_zero_state(handle.value, 3), "zero-state restore failed");
    Fit committed;
    REQUIRE(
        update_l1_host(
            handle.value, kL1Features1, kL1Target1, nullptr, 6, 6.0, l1_config(2, 0.8), committed
        ),
        "baseline L1 update failed"
    );

    RhCudaUpdateConfig unknown = l1_config(2, 0.8);
    unknown.penalty = 2;
    RhCudaUpdateConfig reserved = l1_config(2, 0.8);
    reserved.reserved0 = 1;
    RhCudaUpdateConfig negative = l1_config(2, -0.5);
    RhCudaUpdateConfig not_finite = l1_config(2, std::nan(""));
    const RhCudaUpdateConfig* rejected[] = {&unknown, &reserved, &negative, &not_finite};
    for (const RhCudaUpdateConfig* config : rejected) {
        Fit ignored;
        RhCudaHostBatch batch{};
        initialize(&batch);
        batch.x_design = kL1Features2;
        batch.y = kL1Target2;
        batch.n_rows = 4;
        batch.n_columns = 2;
        batch.batch_weight = 4.0;
        initialize(&ignored.diagnostics);
        REQUIRE(
            rh_cuda_engine_update_host(handle.value, &batch, config, &ignored.diagnostics) ==
                RH_CUDA_STATUS_INVALID_ARGUMENT,
            "a malformed penalty configuration was accepted"
        );
    }

    double coefficients[3] = {};
    double information[9] = {};
    RhCudaHostState state{};
    initialize(&state);
    state.coefficients = coefficients;
    state.information = information;
    REQUIRE(rh_cuda_engine_copy_state(handle.value, &state) == RH_CUDA_STATUS_SUCCESS, "copy");
    REQUIRE(
        std::memcmp(coefficients, committed.coefficients, sizeof(coefficients)) == 0 &&
            std::memcmp(information, committed.information, sizeof(information)) == 0 &&
            state.batch_count == 1 && state.previous_lambda == committed.state.previous_lambda,
        "a rejected update changed the committed state"
    );
    return true;
}

bool case_l1_device_input_matches_host() {
    EngineHandle host_engine;
    EngineHandle device_engine;
    REQUIRE(create_float64_engine(host_engine, 3), "host engine creation failed");
    REQUIRE(create_float64_engine(device_engine, 3), "device engine creation failed");
    REQUIRE(restore_zero_state(host_engine.value, 3), "host zero-state restore failed");
    REQUIRE(restore_zero_state(device_engine.value, 3), "device zero-state restore failed");
    const RhCudaUpdateConfig config = l1_config(2, 0.8);

    Fit host;
    REQUIRE(
        update_l1_host(host_engine.value, kL1Features1, kL1Target1, nullptr, 6, 6.0, config, host),
        "host L1 update failed"
    );

    DeviceBuffer x;
    DeviceBuffer target;
    REQUIRE(x.upload(kL1Features1, 12) && target.upload(kL1Target1, 6), "device upload failed");
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess, "device upload did not complete");
    RhCudaDeviceBatch batch{};
    initialize(&batch);
    batch.x_design = x.value;
    batch.y = target.value;
    batch.n_rows = 6;
    batch.n_columns = 2;
    batch.batch_weight = 6.0;
    Fit device;
    initialize(&device.state);
    initialize(&device.diagnostics);
    device.state.coefficients = device.coefficients;
    device.state.information = device.information;
    REQUIRE(
        rh_cuda_engine_update_device_with_state(
            device_engine.value, &batch, &config, &device.diagnostics, &device.state
        ) == RH_CUDA_STATUS_SUCCESS,
        "device L1 update failed"
    );
    REQUIRE(fits_agree(host, device, 3), "device L1 input changed the fit");
    return true;
}

bool case_prediction_reads_device_input() {
    /*
     * The same fitted engine predicts from a host expanded design, a device
     * expanded design (read in place), and a device feature matrix widened on
     * device.  All three must agree exactly.
     */
    EngineHandle handle;
    Fit fit;
    REQUIRE(create_float64_engine(handle, 2), "engine creation failed");
    REQUIRE(restore_zero_state(handle.value, 2), "zero-state restore failed");
    RhCudaHostBatch batch{};
    initialize(&batch);
    batch.x_design = kWideDesign2;
    batch.y = kTarget;
    batch.n_rows = 4;
    batch.n_columns = 2;
    batch.batch_weight = 4.0;
    const RhCudaUpdateConfig config = unpenalized_config(1);
    initialize(&fit.diagnostics);
    REQUIRE(
        rh_cuda_engine_update_host(handle.value, &batch, &config, &fit.diagnostics) ==
            RH_CUDA_STATUS_SUCCESS,
        "update before prediction failed"
    );

    DeviceBuffer wide;
    DeviceBuffer narrow;
    REQUIRE(wide.upload(kWideDesign2, 8) && narrow.upload(kNarrowDesign2, 4), "upload failed");
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess, "device upload did not complete");

    double from_host[4] = {};
    double from_wide[4] = {};
    double from_narrow[4] = {};
    RhCudaPrediction request{};
    initialize(&request);
    request.n_rows = 4;
    request.n_features_in = 1;

    request.x_design = kWideDesign2;
    request.prediction = from_host;
    request.n_columns = 2;
    request.input_location = RH_CUDA_MEMORY_HOST;
    REQUIRE(rh_cuda_engine_predict(handle.value, &request) == RH_CUDA_STATUS_SUCCESS, "host");

    request.x_design = wide.value;
    request.prediction = from_wide;
    request.input_location = RH_CUDA_MEMORY_DEVICE;
    REQUIRE(rh_cuda_engine_predict(handle.value, &request) == RH_CUDA_STATUS_SUCCESS, "wide");

    request.x_design = narrow.value;
    request.prediction = from_narrow;
    request.n_columns = 1;
    REQUIRE(rh_cuda_engine_predict(handle.value, &request) == RH_CUDA_STATUS_SUCCESS, "narrow");
    REQUIRE(
        std::memcmp(from_host, from_wide, sizeof(from_host)) == 0 &&
            std::memcmp(from_host, from_narrow, sizeof(from_host)) == 0,
        "device prediction differs from host prediction"
    );

    request.x_design = kNarrowDesign2;  /* host memory declared as device */
    REQUIRE(
        rh_cuda_engine_predict(handle.value, &request) != RH_CUDA_STATUS_SUCCESS,
        "a host pointer was accepted as device prediction input"
    );
    request.x_design = narrow.value;
    request.input_location = 7;
    REQUIRE(
        rh_cuda_engine_predict(handle.value, &request) == RH_CUDA_STATUS_INVALID_ARGUMENT,
        "an unknown prediction input location was accepted"
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

    RhCudaUpdateConfig config{};
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
    RhCudaPrediction prediction{};
    initialize(&prediction);
    prediction.x_design = design;
    prediction.prediction = predictions;
    prediction.n_rows = 4;
    prediction.n_columns = 2;
    prediction.n_features_in = 1;
    prediction.input_location = RH_CUDA_MEMORY_HOST;
    if (!check(rh_cuda_engine_predict(engine, &prediction), "rh_cuda_engine_predict")) {
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
        {"l1_stream_matches_reference", case_l1_stream_matches_reference},
        {"l1_never_penalizes_the_intercept", case_l1_never_penalizes_the_intercept},
        {"l1_rejections_leave_state_untouched", case_l1_rejections_leave_state_untouched},
        {"l1_device_input_matches_host", case_l1_device_input_matches_host},
        {"prediction_reads_device_input", case_prediction_reads_device_input},
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
