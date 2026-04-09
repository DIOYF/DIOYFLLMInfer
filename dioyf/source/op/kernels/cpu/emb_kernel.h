//
// Created by zhou on 2026/4/9.
//

#ifndef DIOYF_INFER_EMB_KERNEL_H
#define DIOYF_INFER_EMB_KERNEL_H
#include "tensor/tensor.h"
namespace kernel {
    void emb_kernel_normal(
        const tensor::Tensor& input,
        const tensor::Tensor& weight,
        const tensor::Tensor& output,
        int32_t vocab_size
        );
}  // namespace kernel
#endif //DIOYF_INFER_EMB_KERNEL_H
