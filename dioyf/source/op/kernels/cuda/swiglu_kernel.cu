//
// Created by zhou on 2026/4/12.
//
#include "swiglu_kernel.cuh"

namespace kernel {

    __global__ void swiglu_kernel_cu_fp32(float* input1, float* input2, float* output, int size) {
        int tid = threadIdx.x;
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= size) {return;}
        extern __shared__ float shared_mem[];
        float* smem1 = shared_mem;
        float* smem2 = shared_mem + blockDim.x;

        smem1[tid] = input1[idx];
        smem2[tid] = input2[idx];
        __syncthreads();

        float value = 1.f / (1.f + exp(-smem1[tid]));
        smem1[tid] = value * smem1[tid];

        output[idx] = smem1[tid] * smem2[tid];
    }


    void swiglu_kernel_cu(
        const tensor::Tensor& input1,
        const tensor::Tensor& input2,
        const tensor::Tensor& output,
        void* stream
        ) {
        CHECK_EQ(input1.is_empty(), false);
        CHECK_EQ(input2.is_empty(), false);
        CHECK_EQ(output.is_empty(), false);

        CHECK(input1.device_type() == base::DeviceType::kDeviceCUDA
            && input2.device_type() == base::DeviceType::kDeviceCUDA
            && output.device_type() == base::DeviceType::kDeviceCUDA);

        int size = static_cast<int>(input1.size());
        int threads = 128;
        int blocks = (size + threads - 1) / threads;
        const size_t shmem = threads * sizeof(float) * 2;

        if (stream) {
            cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
            swiglu_kernel_cu_fp32<<<blocks, threads, shmem , stream_>>>(
                const_cast<float*>(input1.ptr<float>()), const_cast<float*>(input2.ptr<float>()), const_cast<float*>(output.ptr<float>()),size);
        }
        else {
            swiglu_kernel_cu_fp32<<<blocks, threads, shmem>>>(
    const_cast<float*>(input1.ptr<float>()), const_cast<float*>(input2.ptr<float>()), const_cast<float*>(output.ptr<float>()),size);
        }


    }


}
