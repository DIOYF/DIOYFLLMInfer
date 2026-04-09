//
// Created by zhou on 2026/4/9.
//

#ifndef DIOYF_INFER_SWIGLU_H
#define DIOYF_INFER_SWIGLU_H

#include "layer.h"
namespace op {
    class SwiGLULayer : public op::Layer {
    public:
        explicit SwiGLULayer(
            base::DeviceType device_type,
            int32_t hidden_dim);

        base::Status check() const override;

        base::Status forward() override;

    private:
        int32_t hidden_dim_ = 0;
    };
}  // namespace op

#endif //DIOYF_INFER_SWIGLU_H
