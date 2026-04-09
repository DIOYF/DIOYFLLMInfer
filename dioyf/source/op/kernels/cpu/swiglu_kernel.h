//
// Created by zhou on 2026/4/9.
//

#ifndef DIOYF_INFER_SWIGLU_KERNEL_H
#define DIOYF_INFER_SWIGLU_KERNEL_H
#include "tensor/tensor.h"
namespace kernel {
    void swiglu_kernel_cpu(
        const tensor::Tensor& input1,
        const tensor::Tensor& input2,
        const tensor::Tensor& output
        );
}
#endif //DIOYF_INFER_SWIGLU_KERNEL_H
