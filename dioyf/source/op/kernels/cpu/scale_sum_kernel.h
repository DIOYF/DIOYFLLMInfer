//
// Created by zhou on 2026/4/9.
//

#ifndef DIOYF_INFER_SCALE_SUM_KERNEL_H
#define DIOYF_INFER_SCALE_SUM_KERNEL_H
#include <tensor/tensor.h>
namespace kernel {
    void scale_sum_kernel_cpu(
        const tensor::Tensor& value,
        const tensor::Tensor& scale,
        const tensor::Tensor& output,
        int t, int d, int stride,
        void* stream = nullptr
        );
}
#endif //DIOYF_INFER_SCALE_SUM_KERNEL_H
