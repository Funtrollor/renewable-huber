#ifndef RENEWABLE_HUBER_RH_CUDA_H
#define RENEWABLE_HUBER_RH_CUDA_H

/*
 * Narrow C ABI for the renewable-huber CUDA engine.
 *
 * This header deliberately contains only fixed-width scalar types, caller
 * owned host buffers, and opaque handles.  It is safe to include from Rust,
 * C, C++, and a Python extension without exposing CUDA, C++, or Rust types
 * across the ABI boundary.  Every public struct begins with abi_version
 * followed by struct_size so future ABI additions can be detected rather
 * than misinterpreted.
 */

#include <stdint.h>

#if defined(RH_CUDA_STATIC)
#  define RH_CUDA_API
#elif defined(_WIN32)
#  if defined(RH_CUDA_BUILD)
#    define RH_CUDA_API __declspec(dllexport)
#  else
#    define RH_CUDA_API __declspec(dllimport)
#  endif
#else
#  define RH_CUDA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define RH_CUDA_ABI_VERSION UINT32_C(1)

typedef int32_t RhCudaStatus;
#define RH_CUDA_STATUS_SUCCESS ((RhCudaStatus)0)
#define RH_CUDA_STATUS_INVALID_ARGUMENT ((RhCudaStatus)1)
#define RH_CUDA_STATUS_UNAVAILABLE ((RhCudaStatus)2)
#define RH_CUDA_STATUS_ALLOCATION_FAILED ((RhCudaStatus)3)
#define RH_CUDA_STATUS_CUDA_ERROR ((RhCudaStatus)4)
#define RH_CUDA_STATUS_CUBLAS_ERROR ((RhCudaStatus)5)
#define RH_CUDA_STATUS_CUSOLVER_ERROR ((RhCudaStatus)6)
#define RH_CUDA_STATUS_NUMERICAL_ERROR ((RhCudaStatus)7)
#define RH_CUDA_STATUS_INTERNAL_ERROR ((RhCudaStatus)8)

typedef int32_t RhCudaDType;
#define RH_CUDA_DTYPE_FLOAT32 ((RhCudaDType)1)
#define RH_CUDA_DTYPE_FLOAT64 ((RhCudaDType)2)

/*
 * Optional engine tuning flags carried in RhCudaEngineOptions::reserved0.
 * Zero preserves the strict, backward-compatible execution contract. CUDA
 * Graph capture is best effort and falls back to ordinary stream launches.
 * FAST_MATH enables TF32 only for float32 dense linear algebra; reductions,
 * convergence decisions, and float64 engines remain strict.
 */
#define RH_CUDA_ENGINE_FLAG_CUDA_GRAPHS (UINT64_C(1) << 0)
#define RH_CUDA_ENGINE_FLAG_FAST_MATH (UINT64_C(1) << 1)
#define RH_CUDA_ENGINE_FLAG_KNOWN_MASK \
    (RH_CUDA_ENGINE_FLAG_CUDA_GRAPHS | RH_CUDA_ENGINE_FLAG_FAST_MATH)

/* Opaque, single-device owner of state, CUDA handles, and reusable buffers. */
typedef struct RhCudaEngine RhCudaEngine;

typedef struct RhCudaEngineOptions {
    uint32_t abi_version;
    uint32_t struct_size;
    RhCudaDType dtype;
    int32_t device_id;
    int64_t n_parameters;
    uint64_t reserved0;
} RhCudaEngineOptions;

/*
 * Caller-owned read-only host state. coefficients is a contiguous vector of
 * n_parameters values and information is a contiguous row-major square
 * n_parameters by n_parameters matrix.  Renewed information is normally
 * symmetric, but a portable checkpoint may contain a general finite matrix;
 * the engine preserves and solves against all of its entries.  The engine
 * explicitly converts this portable row-major layout to and from its internal
 * column-major representation.
 */
typedef struct RhCudaHostStateView {
    uint32_t abi_version;
    uint32_t struct_size;
    const void* coefficients;
    const void* information;
    int64_t n_samples_seen;
    int64_t batch_count;
    double previous_lambda;
    double weight_sum;
} RhCudaHostStateView;

/* Caller-owned writable buffers and metadata for a device-to-host state copy. */
typedef struct RhCudaHostState {
    uint32_t abi_version;
    uint32_t struct_size;
    void* coefficients;
    void* information;
    int64_t n_samples_seen;
    int64_t batch_count;
    double previous_lambda;
    double weight_sum;
} RhCudaHostState;

/* Immutable unpenalized configuration for one update. */
typedef struct RhCudaUnpenalizedConfig {
    uint32_t abi_version;
    uint32_t struct_size;
    int64_t n_features_in;
    int64_t max_iter;
    double tau;
    double bandwidth_scale;
    double tolerance;
    double ridge;
} RhCudaUnpenalizedConfig;

/*
 * Host-fed C-contiguous batch.  x_design is row-major with shape
 * (n_rows, n_parameters); intercept construction belongs to the caller.
 * y and sample_weight, when non-null, are contiguous vectors of n_rows in
 * the engine dtype.  sample_weight has the same frequency-weight semantics
 * as the Python reference.  Passing NULL means all weights are one.
 */
typedef struct RhCudaHostBatch {
    uint32_t abi_version;
    uint32_t struct_size;
    const void* x_design;
    const void* y;
    const void* sample_weight;
    int64_t n_rows;
    int64_t n_columns;
    /* Python validates and computes this scalar before crossing the ABI. */
    double batch_weight;
} RhCudaHostBatch;

/*
 * Device-resident, C-contiguous batch. Pointers must address allocations on
 * the engine's CUDA device and contain values in the engine dtype. The caller
 * (normally a DLPack consumer) retains ownership until this call returns.
 */
typedef struct RhCudaDeviceBatch {
    uint32_t abi_version;
    uint32_t struct_size;
    const void* x_design;
    const void* y;
    const void* sample_weight;
    int64_t n_rows;
    int64_t n_columns;
    double batch_weight;
} RhCudaDeviceBatch;

typedef struct RhCudaHostPrediction {
    uint32_t abi_version;
    uint32_t struct_size;
    const void* x_design;
    void* prediction;
    int64_t n_rows;
    int64_t n_columns;
} RhCudaHostPrediction;

typedef struct RhCudaDiagnostics {
    uint32_t abi_version;
    uint32_t struct_size;
    int64_t iterations;
    int32_t converged;
    int32_t used_regularized_fallback;
    double objective;
    double lambda_value;
    double bandwidth;
} RhCudaDiagnostics;

typedef struct RhCudaRuntimeInfo {
    uint32_t abi_version;
    uint32_t struct_size;
    int32_t runtime_version;
    int32_t driver_version;
    int32_t device_count;
    int32_t reserved0;
} RhCudaRuntimeInfo;

/* Engine-local tuning state and cumulative graph-use counters. */
typedef struct RhCudaEngineFeatures {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t requested_flags;
    uint64_t enabled_flags;
    uint64_t graph_captures;
    uint64_t graph_replays;
    uint64_t graph_fallbacks;
} RhCudaEngineFeatures;

RH_CUDA_API uint32_t rh_cuda_abi_version(void);
/* Thread-local failure text for calls which fail before an engine exists. */
RH_CUDA_API const char* rh_cuda_last_error(void);
RH_CUDA_API RhCudaStatus rh_cuda_is_available(int32_t* available);
RH_CUDA_API RhCudaStatus rh_cuda_device_count(int32_t* count);
RH_CUDA_API RhCudaStatus rh_cuda_runtime_info(RhCudaRuntimeInfo* info);

RH_CUDA_API RhCudaStatus rh_cuda_engine_create(
    const RhCudaEngineOptions* options,
    RhCudaEngine** out_engine
);
RH_CUDA_API RhCudaStatus rh_cuda_engine_destroy(RhCudaEngine* engine);

/* Restore a portable host checkpoint into the engine's persistent device state. */
RH_CUDA_API RhCudaStatus rh_cuda_engine_restore(
    RhCudaEngine* engine,
    const RhCudaHostStateView* state
);

/* Copy persistent device state into caller-owned contiguous host buffers. */
RH_CUDA_API RhCudaStatus rh_cuda_engine_copy_state(
    RhCudaEngine* engine,
    RhCudaHostState* state
);

/* Execute one complete unpenalized Newton batch transition. */
RH_CUDA_API RhCudaStatus rh_cuda_engine_update_host(
    RhCudaEngine* engine,
    const RhCudaHostBatch* batch,
    const RhCudaUnpenalizedConfig* config,
    RhCudaDiagnostics* diagnostics
);

/*
 * Execute one update and export the committed state using the same stream
 * completion.  This avoids a second device synchronization for callers that
 * need a portable checkpoint after every batch.  The caller owns all output
 * buffers, exactly as for rh_cuda_engine_copy_state.
 */
RH_CUDA_API RhCudaStatus rh_cuda_engine_update_host_with_state(
    RhCudaEngine* engine,
    const RhCudaHostBatch* batch,
    const RhCudaUnpenalizedConfig* config,
    RhCudaDiagnostics* diagnostics,
    RhCudaHostState* state
);

/* Return the engine stream handle for DLPack producer/consumer synchronization. */
RH_CUDA_API RhCudaStatus rh_cuda_engine_stream(
    RhCudaEngine* engine,
    uintptr_t* stream
);

/* Device-input equivalent of rh_cuda_engine_update_host_with_state. */
RH_CUDA_API RhCudaStatus rh_cuda_engine_update_device_with_state(
    RhCudaEngine* engine,
    const RhCudaDeviceBatch* batch,
    const RhCudaUnpenalizedConfig* config,
    RhCudaDiagnostics* diagnostics,
    RhCudaHostState* state
);

RH_CUDA_API RhCudaStatus rh_cuda_engine_predict_host(
    RhCudaEngine* engine,
    const RhCudaHostPrediction* request
);

RH_CUDA_API RhCudaStatus rh_cuda_engine_synchronize(RhCudaEngine* engine);

RH_CUDA_API RhCudaStatus rh_cuda_engine_features(
    RhCudaEngine* engine,
    RhCudaEngineFeatures* features
);

/* Returned storage is owned by the engine and stays valid until its next API call. */
RH_CUDA_API const char* rh_cuda_engine_last_error(const RhCudaEngine* engine);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RENEWABLE_HUBER_RH_CUDA_H */
