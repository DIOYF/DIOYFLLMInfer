//
// Created by zhou on 2026/4/9.
//

#include "sampler/argmax_sampler.h"
#include <algorithm>
namespace sampler {
    size_t ArgmaxSampler::sample(const float* logits, size_t size, void* stream) {
        if (device_type_ == base::DeviceType::kDeviceCPU) {
            size_t next = std::distance(logits, std::max_element(logits, logits + size));
            return next;
        } else {
            // todo : cuda
            // return next;
            return 0;
        }
    }
}  // name