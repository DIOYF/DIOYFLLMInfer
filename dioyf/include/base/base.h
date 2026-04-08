#ifndef DIOYF_INFER_BASE_H
#define DIOYF_INFER_BASE_H
#include <glog/logging.h>
#include <cstdint>
#include <string>

#define UNUSED(expr) \
    do { \
        (void)expr; \
    } while(0)

namespace model {

}

namespace base {
    enum class DeviceType : uint8_t {
        kDeviceUnknown = 0,
        kDeviceCPU = 1,
        kDeviceCUDA = 2,
    };

    enum class DataType : uint8_t {
        kDataTypeUnknown = 0,
        kDataTypeFp32 = 1,
        kDataTypeInt8 = 2,
        kDataTypeInt32 = 3,
    };

    enum class ModelType : uint8_t {
        kModelTypeUnknown = 0,
        kModelTypeLLama2 = 1,
    };

    inline size_t DataTypeSize(DataType dataType) {
        if  (dataType == DataType::kDataTypeFp32) {
            return sizeof(float);
        }
        else if (dataType == DataType::kDataTypeInt8) {
            return sizeof(int8_t);
        }
        else if (dataType == DataType::kDataTypeInt32) {
            return sizeof(int32_t);
        }
        else {
            return 0;
        }
    }

    // 定义一个不可复制对象
    class NoCopyable {
    protected:
        NoCopyable() = default;
        ~NoCopyable() = default;

        NoCopyable(const NoCopyable&) = delete;
        NoCopyable& operator=(const NoCopyable&) = delete;
    };



}











#endif //DIOYFLLMINFER_BASE_H
