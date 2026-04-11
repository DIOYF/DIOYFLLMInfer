//
// Created by zhou on 2026/4/9.
//

#ifndef DIOYF_INFER_SCALE_KERNEL_H
#define DIOYF_INFER_SCALE_KERNEL_H
#include <tensor/tensor.h>
namespace kernel {
    void scale_inplace_cpu(float scale,
        const tensor::Tensor& tensor,
        void* stream);
}
#endif //DIOYF_INFER_SCALE_KERNEL_H
