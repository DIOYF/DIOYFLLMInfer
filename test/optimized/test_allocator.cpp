//
// Created by zhou on 2026/4/8.
//

#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "base/buffer.h"
TEST(test_buffer, cpu_allocate) {
    using namespace base;
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    float* ptr = new float[32];
    Buffer buffer(32, nullptr, ptr, true);
    CHECK_EQ(buffer.is_external(), true);
    cudaFree(buffer.ptr());
}


// todo: cuda allocate test