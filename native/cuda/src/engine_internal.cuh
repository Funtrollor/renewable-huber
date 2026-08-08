#ifndef RENEWABLE_HUBER_RH_CUDA_ENGINE_INTERNAL_CUH
#define RENEWABLE_HUBER_RH_CUDA_ENGINE_INTERNAL_CUH

/*
 * Leaf facilities shared by every engine translation unit: the exception type
 * that carries an RhCudaStatus, the CUDA/cuBLAS/cuSOLVER status checks that
 * throw it, the ABI header gate, and raw device allocation.
 *
 * `Failure` must never live in an anonymous namespace here. It is thrown from
 * several translation units and caught in the C API layer; per-TU copies would
 * be distinct types, `catch (const Failure&)` would stop matching, and every
 * error would silently degrade to RH_CUDA_STATUS_INTERNAL_ERROR while the
 * numerics stayed correct. Its destructor is defined out of line, in
 * engine_internal.cu, so the vtable and typeinfo are emitted exactly once.
 */

#include "rh_cuda.h"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

namespace rh_cuda::engine {

// The free functions below are `inline`: this header is included by every
// engine translation unit, and they are small enough that a call is more
// expensive than the body.

using ErrorBuffer = std::array<char, 1024>;


inline void clear_error(ErrorBuffer& buffer) noexcept {
    buffer[0] = '\0';
}

inline void set_error(ErrorBuffer& buffer, const char* message) noexcept {
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

    // Out of line so this class has a key function: the vtable and
    // typeinfo are emitted in one TU instead of merged from many.
    ~Failure() override;

    const char* what() const noexcept override { return message_.c_str(); }
    RhCudaStatus status() const noexcept { return status_; }

private:
    RhCudaStatus status_;
    std::string message_;
};

[[noreturn]] inline void fail(RhCudaStatus status, const std::string& message) {
    throw Failure(status, message);
}

inline std::string with_code(const char* operation, const char* detail) {
    std::ostringstream stream;
    stream << operation << ": " << detail;
    return stream.str();
}

inline void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fail(RH_CUDA_STATUS_CUDA_ERROR, with_code(operation, cudaGetErrorString(status)));
    }
}

inline void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::ostringstream stream;
        stream << operation << ": cuBLAS status " << static_cast<int>(status);
        fail(RH_CUDA_STATUS_CUBLAS_ERROR, stream.str());
    }
}

inline void check_cusolver(cusolverStatus_t status, const char* operation) {
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

inline bool finite_positive(double value) {
    return std::isfinite(value) && value > 0.0;
}

inline bool finite_nonnegative(double value) {
    return std::isfinite(value) && value >= 0.0;
}

inline size_t checked_elements(int64_t rows, int64_t columns, const char* name) {
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
void allocate(
    void** pointer,
    size_t elements,
    const char* name,
    cudaStream_t stream,
    bool stream_ordered,
    cudaMemPool_t memory_pool
) {
    if (elements == 0) {
        *pointer = nullptr;
        return;
    }
    if (elements > std::numeric_limits<size_t>::max() / sizeof(T)) {
        fail(RH_CUDA_STATUS_ALLOCATION_FAILED, std::string(name) + " allocation overflows");
    }
    const cudaError_t status = stream_ordered
        ? cudaMallocFromPoolAsync(pointer, elements * sizeof(T), memory_pool, stream)
        : cudaMalloc(pointer, elements * sizeof(T));
    if (status != cudaSuccess) {
        fail(RH_CUDA_STATUS_ALLOCATION_FAILED, with_code(name, cudaGetErrorString(status)));
    }
}

inline void release(void*& pointer, cudaStream_t stream, bool stream_ordered) noexcept {
    if (pointer != nullptr) {
        if (stream_ordered && stream != nullptr) {
            cudaFreeAsync(pointer, stream);
        } else {
            cudaFree(pointer);
        }
        pointer = nullptr;
    }
}

/// One process-wide stream-ordered memory pool per device.
cudaMemPool_t shared_device_pool(int device_id);

}  // namespace rh_cuda::engine

#endif  // RENEWABLE_HUBER_RH_CUDA_ENGINE_INTERNAL_CUH
