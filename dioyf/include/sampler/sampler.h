//
// Created by zhou on 2026/4/9.
//

#ifndef DIOYF_INFER_SAMPLER_H
#define DIOYF_INFER_SAMPLER_H
#include <cstddef>
#include <cstdint>
#include "base/base.h"

namespace sampler {
    class Sampler {
    public:
        explicit Sampler(base::DeviceType device_type) : device_type_(device_type) {}

        virtual ~Sampler() {}

        virtual size_t sample(const float* logits, size_t size, void* stream = nullptr) = 0;

    protected:
        base::DeviceType device_type_;
    };
}
// namespace sampler
#endif //DIOYF_INFER_SAMPLER_H
