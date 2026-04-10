//
// Created by zhou on 2026/4/9.
//

#ifndef DIOYF_INFER_CONFIG_H
#define DIOYF_INFER_CONFIG_H
#include <cstdint>

namespace model {

    enum class ModelBufferType {
        kInputTokens = 0,
        kInputEmbeddings = 1,
        kOutputRMSNorm = 2,
        kKeyCache = 3,
        kValueCache = 4,
        kQuery = 5,
        kInputPos = 6,
        kScoreStorage = 7,
        kOutputMHA = 8,
        kAttnOutput = 9,
        kW1Output = 10,
        kW2Output = 11,
        kW3Output = 12,
        kFFNRMSNorm = 13,
        kForwardOutput = 15,
        kForwardOutputCPU = 16,

        kSinCache = 17,
        kCosCache = 18,
      };

    struct ModelConfig {
        int32_t dim = 0;
        int32_t hidden_dim = 0;
        int32_t layer_num = 0;
        int32_t head_num = 0;
        int32_t kv_head_num = 0;
        int32_t vocab_size = 0;
        int32_t seq_len = 0;
    };

    struct TransformerConfig {
        int32_t kv_dim_ = 0;
        int32_t kv_mul_ = 0;
        int32_t head_size_ = 0;
        int32_t vocab_size_ = 0;

        int32_t dim_ = 0;
        int32_t hidden_dim_ = 0;
        int32_t layer_num_ = 0;
        int32_t head_num_ = 0;
        int32_t kv_head_num_ = 0;
        int32_t seq_len_ = 0;
        bool is_shared_weight_ = false;
    };

}



#endif //DIOYF_INFER_CONFIG_H
