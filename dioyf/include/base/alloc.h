#ifndef DIOYF_INFER_ALLOC_H
#define DIOYF_INFER_ALLOC_H

#include <map>
#include <memory>
#include "base.h"

namespace base {
    enum class MemcpyKind {
        kMemcpyCPU2CPU = 0,
        kMemcpyCPU2CUDA =1,
        kMemcpyCUDA2CPU =2,
        kMemcpyCUDA2CUDA =3,
    };


    class DeviceAllocator {
    public:
        explicit DeviceAllocator(DeviceType deviceType) : device_type_(deviceType) {}

        virtual ~DeviceAllocator() {}

        virtual DeviceType device_type() const {return device_type_;}

        virtual void release(void* ptr) const = 0;

        virtual void* allocate(size_t byte_size) const = 0;

        virtual void memcpy(const void* src_ptr, void* dest_ptr,
            size_t byte_size,
            MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU,
            void* stream = nullptr, bool need_sync = false ) const;

        virtual void memset_zero(void* ptr, size_t byte_size, void* stream = nullptr, bool need_sync = false) const;

    private:
        DeviceType device_type_ = DeviceType::kDeviceUnknown;
    };


    class CPUDeviceAllocator : public DeviceAllocator {
    public:
        explicit CPUDeviceAllocator();

        void* allocate(size_t byte_size) const override;

        void release(void* ptr) const override;

    };

    // todo:CUDA allocator

    class CUDADeviceAllocator : public DeviceAllocator {
    public:
        explicit CUDADeviceAllocator();
    };


    class CPUDeviceAllocatorFactory {
    public:
        static std::shared_ptr<CPUDeviceAllocator> get_instance() {
            if (instance == nullptr) {
                instance = std::make_shared<CPUDeviceAllocator>();
            }
            return instance;
        }
    private:
        static std::shared_ptr<CPUDeviceAllocator> instance;;
    };









}
























#endif //DIOYF_INFER_ALLOC_H
