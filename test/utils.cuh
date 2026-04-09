#ifndef TEST_CU_CUH
#define TEST_CU_CUH

void test_function(float* arr, int32_t size, float value = 1.f);

void set_value_cu(float* arr_cu, int32_t size, float value = 1.f);

void set_value_cpu(float* arr_cpu, int32_t size, float value = 1.f);

#endif