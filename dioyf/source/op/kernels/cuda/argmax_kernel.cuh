//
// Created by zhou on 2026/4/12.
//

#ifndef DIOYF_INFER_ARGMAX_KERNEL_CUH
#define DIOYF_INFER_ARGMAX_KERNEL_CUH
namespace kernel {
    size_t argmax_kernel_cu(const float* input_ptr, size_t size, void* stream);
}
#endif //DIOYF_INFER_ARGMAX_KERNEL_CUH
