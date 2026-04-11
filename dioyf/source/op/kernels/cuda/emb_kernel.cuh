//
// Created by zhou on 2026/4/11.
//

#ifndef DIOYF_INFER_EMB_KERNEL_CUH
#define DIOYF_INFER_EMB_KERNEL_CUH
#include "tensor/tensor.h"
namespace kernel {
    void emb_kernel_cu(
        const tensor::Tensor& input,
        const tensor::Tensor& weight,
        const tensor::Tensor& output,
        int32_t vocab_size,
        void* stream = nullptr
        );
}  // namespace kernel
#endif //DIOYF_INFER_EMB_KERNEL_CUH
