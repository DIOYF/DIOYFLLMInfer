//
// Created by zhou on 2026/4/11.
//

#ifndef DIOYF_INFER_SWIGLU_KERNEL_CUH
#define DIOYF_INFER_SWIGLU_KERNEL_CUH
#include "tensor/tensor.h"
namespace kernel {
    void swiglu_kernel_cu(
            const tensor::Tensor& input1,
            const tensor::Tensor& input2,
            const tensor::Tensor& output,
            void* stream = nullptr
            ) ;
}
#endif //DIOYF_INFER_SWIGLU_KERNEL_CUH
