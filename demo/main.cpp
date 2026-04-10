//
// Created by zhou on 2026/4/9.
//

#include "base/base.h"
#include "glog/logging.h"
#include "model/llama.h"

int32_t generate(
    const model::LLama2Model& model,
    const std::string& sentence,
    int total_steps,
    bool need_output = false) {

    auto tokens = model.encode(sentence);

    int32_t prompt_len = tokens.size();

    LOG_IF(FATAL, tokens.empty()) << "The token is empty";

    int32_t pos = 0;
    int32_t next = -1;
    bool is_prompt = true;

    const auto& prompt_embedding = model.embedding(tokens);
    tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

    std::vector<int32_t> words;
    while (pos < total_steps) {
        pos_tensor.index<int32_t>(0) = pos;
        if (pos < prompt_len -1) {
            tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
            model.predict(input, pos_tensor, is_prompt, next);
        }
        else {
            is_prompt = false;
            tokens = std::vector<int32_t>{next};
            const auto& token_embedding = model.embedding(tokens);
            tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
            model.predict(input, pos_tensor, is_prompt, next);
        }

        if (model.is_sentence_ending(next)) {
            break;
        }
        if (is_prompt) {
            next = tokens.at(pos + 1);
            words.push_back(next);
        }
        else {
            words.push_back(next);
        }

        pos++;
    }

    if (need_output) {
        printf("%s\n", model.decode(words).c_str());
        fflush(stdout);
    }
    return std::min(pos, total_steps);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        LOG(INFO) << "Usage: ./demo checkpoint path tokenizer path";
        LOG(INFO) << "please use ./llama_infer /models/llama_model_1B.bin /models/tokenizer.model ";
        return -1;
    }

    const char* checkpoint_path = argv[1];
    const char* tokenizer_path = argv[2];

    model::LLama2Model model(base::TokenizerType::kEncodeSpe,
        tokenizer_path, checkpoint_path, false);

    auto init_status = model.init(base::DeviceType::kDeviceCPU);

    if (!init_status) {
        LOG(FATAL) << "The model init failed, the error code is: " << init_status.get_err_code();
    }

    std::string sentence = " ";
    printf(" input prompt: \n");
    std::cin >> sentence;

    printf(" Generate result: \n");
    auto start_time = std::chrono::steady_clock::now();
    fflush(stdout);

    int steps = generate(model, sentence, 128, true);

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(end_time - start_time);
    printf("\n infer time steps per second : %lf \n ", steps / duration.count());
    fflush(stdout);
    return 0;
}
