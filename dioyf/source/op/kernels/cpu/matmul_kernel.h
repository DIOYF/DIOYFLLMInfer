//
// Created by zhou on 2026/4/9.
//

#ifndef DIOYF_INFER_MATMUL_KERNEL_H
#define DIOYF_INFER_MATMUL_KERNEL_H

#include "tensor/tensor.h"
namespace kernel {
    void matmul_kernel_cpu(
        const tensor::Tensor& input,
        const tensor::Tensor& weight,
        const tensor::Tensor& output,
        float scale = 1.f
        );
}  // namespace kernel

#endif //DIOYF_INFER_MATMUL_KERNEL_H
