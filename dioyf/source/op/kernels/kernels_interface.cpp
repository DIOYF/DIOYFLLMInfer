//
// Created by zhou on 2026/4/9.
//

#include "kernels_interface.h"
#include "cpu/rmsnorm_kernel.h"

namespace kernel {

    RMSNormKernel RMSNorm(base::DeviceType device_type) {
        if (device_type == base::DeviceType::kDeviceCPU) {
            return rmsnorm_kernel_cpu;
        }
        else if (device_type == base::DeviceType::kDeviceCUDA) {
            // todo: cuda compute
            return nullptr;
        }
        else {
            LOG(FATAL) << "Unknown device type for get a rmsnorm kernel.";
            return nullptr;
        }
        return nullptr;
    }

    RMSNormKernelDim RMSNormDim(base::DeviceType device_type) {
        if (device_type == base::DeviceType::kDeviceCUDA) {
            // todo: cuda compute
            return nullptr;
        } else{
            LOG(FATAL) << "Unknown device type for get a rmsnorm dim kernel.";
            return nullptr;
        }
        return nullptr;

    }



}

