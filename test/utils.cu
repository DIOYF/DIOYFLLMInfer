#include <glog/logging.h>
#include "utils.cuh"


void test_function(float *arr, int32_t size, float value) {
    // todo cuda test
}

void set_value_cu(float* arr_cu, int32_t size, float value) {
    // todo cuda test
}

void set_value_cpu(float* arr_cpu, int32_t size, float value) {
    // cpu value
    for (int i=0; i<size; i++) {
        arr_cpu[i] = value;
    }
}

