//
// Created by zhou on 2026/4/9.
//

#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <tensor/tensor.h>
#include "../utils.cuh"
#include "base/buffer.h"


TEST(test_tensor, to_cpu) {
    using namespace base;
    auto alloc_cu = CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor t1_cu(DataType::kDataTypeFp32, 32, 32, true, alloc_cu);
    ASSERT_EQ(t1_cu.is_empty(), false);
    set_value_cpu(t1_cu.ptr<float>(), 32 * 32);

    t1_cu.to_cpu();
    ASSERT_EQ(t1_cu.device_type(), base::DeviceType::kDeviceCPU);
    float* cpu_ptr = t1_cu.ptr<float>();
    for (int i = 0; i < 32 * 32; ++i) {
        ASSERT_EQ(*(cpu_ptr + i), 1.f);
    }
}


TEST(test_tensor, clone_cpu) {
    using namespace base;
    auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor t1_cpu(DataType::kDataTypeFp32, 32, 32, true, alloc_cpu);
    ASSERT_EQ(t1_cpu.is_empty(), false);
    for (int i = 0; i < 32 * 32; ++i) {
        t1_cpu.index<float>(i) = 3.f;
    }

    tensor::Tensor t2_cpu = t1_cpu.clone();
    float* p2 = new float[32 * 32];
    std::memcpy(p2, t2_cpu.ptr<float>(), sizeof(float) * 32 * 32);
    for (int i = 0; i < 32 * 32; ++i) {
        ASSERT_EQ(p2[i], 3.f);
    }

    std::memcpy(p2, t1_cpu.ptr<float>(), sizeof(float) * 32 * 32);
    for (int i = 0; i < 32 * 32; ++i) {
        ASSERT_EQ(p2[i], 3.f);
    }
    delete[] p2;
}


TEST(test_tensor, init_cpu) {
    using namespace base;
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

    int32_t size = 32 * 151;

    tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
    ASSERT_EQ(t1.is_empty(), false);
}

TEST(test_tensor, assign_cpu) {
    using namespace base;
    auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor t1_cpu(DataType::kDataTypeFp32, 32, 32, true, alloc_cpu);
    ASSERT_EQ(t1_cpu.is_empty(), false);

    int32_t size = 32 * 32;
    float* ptr = new float[size];
    for (int i = 0; i < size; ++i) {
        ptr[i] = float(i);
    }
    std::shared_ptr<Buffer> buffer =
        std::make_shared<Buffer>(size * sizeof(float), nullptr, ptr, true);
    buffer->set_device_type(DeviceType::kDeviceCPU);

    ASSERT_EQ(t1_cpu.assign(buffer), true);
    ASSERT_EQ(t1_cpu.is_empty(), false);
    ASSERT_NE(t1_cpu.ptr<float>(), nullptr);
    delete[] ptr;
}


// todo cuda test

