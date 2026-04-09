//
// Created by zhou on 2026/4/9.
//

#ifndef DIOYF_INFER_ADD_H
#define DIOYF_INFER_ADD_H
#include "base/base.h"
#include "layer.h"


namespace op {
    class VecAddLayer : public Layer {
    public:
        explicit VecAddLayer(base::DeviceType device_type);

        base::Status check() const override;

        base::Status forward() override;
    };
}



#endif //DIOYF_INFER_ADD_H
