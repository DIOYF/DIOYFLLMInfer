//
// Created by zhou on 2026/4/9.
//

#ifndef DIOYF_INFER_ARGMAX_SAMPLER_H
#define DIOYF_INFER_ARGMAX_SAMPLER_H

#include "sampler.h"
namespace sampler {
    class ArgmaxSampler : public Sampler {
    public:
        explicit ArgmaxSampler(base::DeviceType device_type) : Sampler(device_type) {}

        size_t sample(const float* logits, size_t size, void* stream) override;
    };
}  // namespace sampler
#endif //DIOYF_INFER_ARGMAX_SAMPLER_H
