//
// Created by zhou on 2026/4/11.
//
#include "add_kernel.cuh"

namespace kernel {
    __global__ void add_kernel_cu_fp32(
        const float * input1, const float* input2,
        float * output, int size) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid >= size) {
            return;
        }

        float in_val1 = input1[tid];
        float in_val2 = input2[tid];
        output[tid] = in_val1 + in_val2;
    }



    void add_kernel_cu(
        const tensor::Tensor& input1,
        const tensor::Tensor& input2,
        const tensor::Tensor& output,
        void* stream) {

        CHECK_EQ(input1.is_empty(), false);
        CHECK_EQ(input2.is_empty(), false);
        CHECK_EQ(output.is_empty(), false);

        int32_t size = static_cast<int32_t>(input1.size());
        CHECK_EQ(size, input2.size());
        CHECK_EQ(size, output.size());

        int32_t thread_num = 512;
        int32_t block_num = (size + thread_num - 1)/thread_num;

        if (stream) {
            cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
            add_kernel_cu_fp32<<<block_num, thread_num, 0, stream_>>>(input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()), size);
        }
        else {
            add_kernel_cu_fp32<<<block_num, thread_num>>>(input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()), size);
        }
    }
}