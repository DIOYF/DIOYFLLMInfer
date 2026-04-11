//
// Created by zhou on 2026/4/11.
//

#ifndef DIOYF_INFER_MATMUL_KERNEL_CUH
#define DIOYF_INFER_MATMUL_KERNEL_CUH
#include "tensor/tensor.h"
#include "base/cuda_config.h"

namespace kernel {
    void matmul_kernel_cu(
        const tensor::Tensor& input,
        const tensor::Tensor& weight,
        const tensor::Tensor& output,
        float scale = 1.f,
        const CudaConfig* config = nullptr
        );

    void matmul_kernel_cu_qint8(
        const tensor::Tensor& input,
        const tensor::Tensor& weight,
        const tensor::Tensor& output, int32_t group_size,
        const tensor::Tensor& scale,
        const CudaConfig* config = nullptr
        );
}  // namespace kernel
#endif //DIOYF_INFER_MATMUL_KERNEL_CUH
