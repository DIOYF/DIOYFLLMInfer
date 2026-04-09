//
// Created by zhou on 2026/4/9.
//

#ifndef DIOYF_INFER_RMSNORM_KERNEL_H
#define DIOYF_INFER_RMSNORM_KERNEL_H

#include "tensor/tensor.h"

namespace kernel {
    void rmsnorm_kernel_cpu(const tensor::Tensor& input,
        const tensor::Tensor& weight,
        const tensor::Tensor& output);
}

#endif //DIOYF_INFER_RMSNORM_KERNEL_H
