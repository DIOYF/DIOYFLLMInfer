#include <glog/logging.h>
#include <gtest/gtest.h>
#include <model/llama.h>
#include <tensor/tensor.h>
#include "base/buffer.h"
#include "base/base.h"

TEST(test_llama_model, cpu1) {
    using namespace base;
    std::shared_ptr<base::CPUDeviceAllocator> alloc =
    std::make_shared<base::CPUDeviceAllocator>();

    const char* checkpoint_path = "../../models/stories110M.bin";  // e.g. out/model.bin
    const char* tokenizer_path = "../../models/tokenizer.model";
    model::LLama2Model model(base::TokenizerType::kEncodeSpe,tokenizer_path, checkpoint_path, false);
    auto status = model.init(base::DeviceType::kDeviceCPU);
    if (status) {
        std::string sentence = "Hi";  // prompts
        const auto& tokens = model.encode(sentence);
        const auto& prompt_embedding = model.embedding(tokens);
        tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);
        tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, false);
        int next;
        const auto s = model.forward(input, pos_tensor, next);
        const float* logits =
        model.get_buffer(model::ModelBufferType::kForwardOutput).ptr<float>();
        LOG(INFO) << logits[0] << logits[32] << logits[128];

        // ASSERT_NEAR(logits[0], -12.7976265, 1e-3f);
        // ASSERT_NEAR(logits[32], -9.97821331, 1e-3f);
        // ASSERT_NEAR(logits[128], -12.8054199, 1e-3f);
        // ASSERT_NEAR(logits[256], -12.7876959, 1e-3f);
        // ASSERT_NEAR(logits[512], 4.75685883, 1e-3f);
        // ASSERT_NEAR(logits[613], -3.83690214, 1e-3f);
        // ASSERT_NEAR(logits[1011], -3.34461427, 1e-3f);
        // ASSERT_NEAR(logits[1022], -7.45470142, 1e-3f);
        // ASSERT_NEAR(logits[1023], -1.00463259, 1e-3f);
  }
}