//
// Created by zhou on 2026/4/9.
//

#ifndef DIOYF_INFER_RMSNORM_H
#define DIOYF_INFER_RMSNORM_H

#include "layer.h"

namespace op {
    class RMSNormLayer : public LayerParam {
    public:
        explicit RMSNormLayer(base::DeviceType device_type, int32_t dim);

        base::Status check() const override;

        base::Status forward() override;

    private:
        int32_t dim_ = 0;
    };
}


#endif //DIOYF_INFER_RMSNORM_H
