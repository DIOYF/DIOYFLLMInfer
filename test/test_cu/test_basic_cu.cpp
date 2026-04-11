//
// Created by zhou on 2026/4/11.
//

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <armadillo>
#include <cuda_runtime_api.h>

#include "../utils.cuh"

TEST(test_cu, test_function) {
    int32_t size = 32;
    float* ptr = new float[size];

    test_function(ptr, size, 2.5f);

    for (int32_t i = 0; i < size; i++) {
        ASSERT_EQ(ptr[i] , 2.5f);
    }

    delete[] ptr;
}


TEST(test_cu, test_set_cu) {
    int32_t size = 32;
    float* ptr = new float[size];
    float* arr_cu = nullptr;
    cudaMalloc((void**)&arr_cu, size * sizeof(float));

    set_value_cu(arr_cu, size, 2.5f);
    delete[] ptr;
    cudaFree(arr_cu);
}
