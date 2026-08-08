#include "rh_cuda.h"

#include "huber_kernels.cuh"

#include "blas_traits.cuh"
#include "engine_internal.cuh"
#include "engine_state.cuh"
#include "pipeline.cuh"
#include "linear_solver.cuh"
#include "objective.cuh"
#include "batch.cuh"
#include "workspace.cuh"

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
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace rh_cuda::engine;


namespace {

thread_local ErrorBuffer g_last_error{};

template <typename T>
bool validate_finite_state(
    const T* coefficients,
    const T* information,
    size_t parameters
) {
    for (size_t index = 0; index < parameters; ++index) {
        if (!std::isfinite(static_cast<double>(coefficients[index]))) {
            fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "restored coefficients must be finite");
        }
    }

    bool symmetric = true;
    for (size_t row = 0; row < parameters; ++row) {
        for (size_t column = 0; column < parameters; ++column) {
            const T value = information[row * parameters + column];
            if (!std::isfinite(static_cast<double>(value))) {
                fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "restored information must be finite");
            }
            if (column < row) {
                const T transpose = information[column * parameters + row];
                // A portable checkpoint is allowed to contain a general dense
                // information matrix. Even a small asymmetry must retain the
                // established full-matrix LU semantics.
                symmetric = symmetric && value == transpose;
            }
        }
    }
    return symmetric;
}



}  // namespace


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
    if ((options->reserved0 & ~RH_CUDA_ENGINE_FLAG_KNOWN_MASK) != 0) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "engine options contain unknown tuning flags");
    }
    if ((options->reserved0 & RH_CUDA_ENGINE_FLAG_FAST_MATH) != 0 &&
        options->dtype != RH_CUDA_DTYPE_FLOAT32) {
        fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "fast math is supported only by float32 engines");
    }
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
        engine->requested_flags = options->reserved0;
        engine->enabled_flags = options->reserved0;
        int memory_pools_supported = 0;
        check_cuda(
            cudaDeviceGetAttribute(
                &memory_pools_supported,
                cudaDevAttrMemoryPoolsSupported,
                options->device_id
            ),
            "query stream-ordered allocation support"
        );
        engine->stream_ordered_allocations = memory_pools_supported != 0;
        if (engine->stream_ordered_allocations) {
            // Use a library-owned pool so retention never mutates CUDA's
            // process-wide default allocator or another framework's policy.
            engine->memory_pool = shared_device_pool(options->device_id);
        }
        check_cuda(cudaStreamCreateWithFlags(&engine->stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        check_cublas(cublasCreate(&engine->cublas), "cublasCreate");
        check_cublas(cublasSetStream(engine->cublas, engine->stream), "cublasSetStream");
        check_cublas(cublasSetPointerMode(engine->cublas, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
        const cublasMath_t dense_math_mode =
            (engine->enabled_flags & RH_CUDA_ENGINE_FLAG_FAST_MATH) != 0
            ? CUBLAS_TF32_TENSOR_OP_MATH
            : CUBLAS_PEDANTIC_MATH;
        check_cublas(cublasSetMathMode(engine->cublas, dense_math_mode), "cublasSetMathMode");
        check_cublas(cublasCreate(&engine->cublas_reduction), "cublasCreate reduction handle");
        check_cublas(
            cublasSetStream(engine->cublas_reduction, engine->stream),
            "cublasSetStream reduction handle"
        );
        check_cublas(
            cublasSetPointerMode(engine->cublas_reduction, CUBLAS_POINTER_MODE_DEVICE),
            "cublasSetPointerMode reduction handle"
        );
        check_cublas(
            cublasSetMathMode(engine->cublas_reduction, CUBLAS_PEDANTIC_MATH),
            "cublasSetMathMode reduction handle"
        );
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
            engine->information_is_symmetric = validate_finite_state(
                  static_cast<const float*>(state->coefficients),
                  static_cast<const float*>(state->information),
                  parameters
            );
        } else {
            engine->information_is_symmetric = validate_finite_state(
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
        if (engine->dtype == RH_CUDA_DTYPE_FLOAT32) {
            enqueue_state_copy<float>(
                engine,
                typed<float>(engine->d_coefficients),
                typed<float>(engine->d_information),
                state
            );
        } else {
            enqueue_state_copy<double>(
                engine,
                typed<double>(engine->d_coefficients),
                typed<double>(engine->d_information),
                state
            );
        }
        check_cuda(cudaStreamSynchronize(engine->stream), "complete state copy");
        fill_state_metadata(engine, state);
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
        check_header(batch, "host batch");
        const BatchView view{
            batch->x_design,
            batch->y,
            batch->sample_weight,
            batch->n_rows,
            batch->n_columns,
            batch->batch_weight,
            cudaMemcpyHostToDevice,
        };
        if (engine->dtype == RH_CUDA_DTYPE_FLOAT32) {
            return update_typed<float>(engine, view, config, diagnostics);
        }
        return update_typed<double>(engine, view, config, diagnostics);
    });
}

RhCudaStatus rh_cuda_engine_update_host_with_state(
    RhCudaEngine* engine,
    const RhCudaHostBatch* batch,
    const RhCudaUnpenalizedConfig* config,
    RhCudaDiagnostics* diagnostics,
    RhCudaHostState* state
) {
    return guarded(engine, [&]() -> RhCudaStatus {
        check_header(batch, "host batch");
        const BatchView view{
            batch->x_design,
            batch->y,
            batch->sample_weight,
            batch->n_rows,
            batch->n_columns,
            batch->batch_weight,
            cudaMemcpyHostToDevice,
        };
        if (engine->dtype == RH_CUDA_DTYPE_FLOAT32) {
            return update_typed<float>(engine, view, config, diagnostics, state);
        }
        return update_typed<double>(engine, view, config, diagnostics, state);
    });
}

RhCudaStatus rh_cuda_engine_stream(RhCudaEngine* engine, uintptr_t* stream) {
    return guarded(engine, [&]() -> RhCudaStatus {
        if (stream == nullptr) {
            fail(RH_CUDA_STATUS_INVALID_ARGUMENT, "stream output must not be null");
        }
        *stream = reinterpret_cast<uintptr_t>(engine->stream);
        return RH_CUDA_STATUS_SUCCESS;
    });
}

RhCudaStatus rh_cuda_engine_update_device_with_state(
    RhCudaEngine* engine,
    const RhCudaDeviceBatch* batch,
    const RhCudaUnpenalizedConfig* config,
    RhCudaDiagnostics* diagnostics,
    RhCudaHostState* state
) {
    return guarded(engine, [&]() -> RhCudaStatus {
        check_header(batch, "device batch");
        const BatchView view{
            batch->x_design,
            batch->y,
            batch->sample_weight,
            batch->n_rows,
            batch->n_columns,
            batch->batch_weight,
            cudaMemcpyDeviceToDevice,
        };
        if (engine->dtype == RH_CUDA_DTYPE_FLOAT32) {
            return update_typed<float>(engine, view, config, diagnostics, state);
        }
        return update_typed<double>(engine, view, config, diagnostics, state);
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

RhCudaStatus rh_cuda_engine_features(
    RhCudaEngine* engine,
    RhCudaEngineFeatures* features
) {
    return guarded(engine, [&]() -> RhCudaStatus {
        check_header(features, "engine features");
        features->requested_flags = engine->requested_flags;
        features->enabled_flags = engine->enabled_flags;
        features->graph_captures = engine->graph_captures;
        features->graph_replays = engine->graph_replays;
        features->graph_fallbacks = engine->graph_fallbacks;
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
