//
// Created by zhou on 2026/4/9.
//

#ifndef DIOYF_INFER_KERNELS_INTERFACE_H
#define DIOYF_INFER_KERNELS_INTERFACE_H

#include "tensor/tensor.h"

namespace kernel {

    typedef void (*RMSNormKernel) (
        const tensor::Tensor& input,
        const tensor::Tensor& weight,
        const tensor::Tensor& output
        );

    typedef void (*RMSNormKernelDim) (
        const tensor::Tensor& input,
        const tensor::Tensor& weight,
        const tensor::Tensor& output,
        int32_t dim);


    RMSNormKernel RMSNorm(base::DeviceType device_type);

    RMSNormKernelDim RMSNormDim(base::DeviceType device_type);


}





#endif //DIOYF_INFER_KERNELS_INTERFACE_H
