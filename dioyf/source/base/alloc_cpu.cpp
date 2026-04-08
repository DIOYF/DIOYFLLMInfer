//
// Created by zhou on 2026/4/8.
//

#include "base/alloc.h"
#include "glog/logging.h"
#include <cstdlib>

#if (defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO >= 200112L))
#define DIOYF_HAVE_POSIX_MEMALIGN
#endif

namespace base {
    CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {
    }

    void* CPUDeviceAllocator::allocate(size_t byte_size) const {
        if (!byte_size) {
            return nullptr;
        }


#ifdef DIOYF_HAVE_POSIX_MEMALIGN
        // 申请内存对齐，按16位或32位对齐申请
        void* data = nullptr;
        const size_t alignment = (byte_size >= size_t(1024)) ? size_t(32) : size_t(16);
        int status = posix_memalign((void**)&data,
            ((alignment >= sizeof(void*)) ? alignment : sizeof(void*)),
            byte_size);
#else
        void * data = malloc(byte_size);
        return data;
#endif
    }

    void CPUDeviceAllocator::release(void* ptr) const {
        if (ptr) {
            free(ptr);
        }
    }

    std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::instance = nullptr;

}