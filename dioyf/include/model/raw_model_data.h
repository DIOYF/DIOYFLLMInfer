//
// Created by zhou on 2026/4/9.
//

#ifndef DIOYF_INFER_RAW_MODEL_DATA_H
#define DIOYF_INFER_RAW_MODEL_DATA_H
#include <cstddef>
#include <cstdint>

namespace model {
    struct RawModelData {
        virtual ~RawModelData();
        int32_t fd = -1;
        size_t file_size = 0;
        void* data = nullptr;
        void* weight_data = nullptr;

      virtual const void* weight(size_t offset) const = 0;
    };

    struct RawModelDataFp32 : public RawModelData {
        const void * weight(size_t offset) const override;
    };

    struct RawModelDataInt8 : public RawModelData {
        const void * weight(size_t offset) const override;
    };

}


#endif //DIOYF_INFER_RAW_MODEL_DATA_H
