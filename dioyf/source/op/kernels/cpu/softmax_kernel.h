//
// Created by zhou on 2026/4/9.
//

#ifndef DIOYF_INFER_SOFTMAX_KERNEL_H
#define DIOYF_INFER_SOFTMAX_KERNEL_H
#include "tensor/tensor.h"
namespace kernel {
    void softmax_inplace_cpu(
        const tensor::Tensor& input,
        void* stream = nullptr
        );
}  // namespace kernel
#endif //DIOYF_INFER_SOFTMAX_KERNEL_H
