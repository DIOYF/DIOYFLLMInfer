//
// Created by zhou on 2026/4/9.
//

#ifndef DIOYF_INFER_CUDA_CONFIG_H
#define DIOYF_INFER_CUDA_CONFIG_H

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
namespace kernel {
    struct CudaConfig {
        cudaStream_t stream = nullptr;
        ~CudaConfig() {
            if (stream) {
                cudaStreamDestroy(stream);
            }
        }
    };
}  // namespace kernel
#endif //DIOYF_INFER_CUDA_CONFIG_H
