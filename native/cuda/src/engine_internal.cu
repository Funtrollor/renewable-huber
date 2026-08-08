#include "engine_internal.cuh"

#include <cuda_runtime_api.h>

#include <mutex>
#include <unordered_map>

namespace rh_cuda::engine {

// Key function for the exception type: emitting the vtable and typeinfo in
// exactly one TU keeps `catch (const Failure&)` matching once the engine spans
// several of them.
Failure::~Failure() = default;

namespace {

class DevicePoolRegistry final {
public:
    cudaMemPool_t acquire(int device_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto existing = pools_.find(device_id);
        if (existing != pools_.end()) {
            return existing->second;
        }
        cudaMemPoolProps properties{};
        properties.allocType = cudaMemAllocationTypePinned;
        properties.handleTypes = cudaMemHandleTypeNone;
        properties.location.type = cudaMemLocationTypeDevice;
        properties.location.id = device_id;
        cudaMemPool_t pool = nullptr;
        check_cuda(cudaMemPoolCreate(&pool, &properties), "create renewable CUDA memory pool");
        uint64_t release_threshold = uint64_t{1} << 30;
        check_cuda(
            cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &release_threshold),
            "configure renewable CUDA memory-pool release threshold"
        );
        pools_.emplace(device_id, pool);
        return pool;
    }

    ~DevicePoolRegistry() noexcept {
        for (const auto& [device_id, pool] : pools_) {
            cudaSetDevice(device_id);
            cudaMemPoolDestroy(pool);
        }
    }

private:
    std::mutex mutex_;
    std::unordered_map<int, cudaMemPool_t> pools_;
};

}  // namespace

cudaMemPool_t shared_device_pool(int device_id) {
    static DevicePoolRegistry registry;
    return registry.acquire(device_id);
}

}  // namespace rh_cuda::engine
