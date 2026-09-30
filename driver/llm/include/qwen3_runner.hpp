#pragma once

#include <functional>
#include <memory>
#include <string>

namespace qwen3 {

enum class ThinkingMode {
    kDefault,
    kEnabled,
    kDisabled,
};

struct ModelLoadOptions {
    std::string model_dir;
    std::string system_prompt = "You are a helpful assistant.";
    bool dynamic_load = false;
    int dynamic_load_pool_size = 2;
    bool memory_guard = true;
    int memory_guard_floor_mb = 128;
};

struct GenerationOptions {
    int max_tokens = 128;
    float temperature = 0.0F;
    float top_p = 1.0F;
    int top_k = 0;
    float repetition_penalty = 1.0F;
    float frequency_penalty = 0.0F;
    float presence_penalty = 0.0F;
    ThinkingMode thinking = ThinkingMode::kDisabled;
    bool reset_context = true;
    std::function<void(const std::string&)> on_token;
};

struct InferenceResult {
    std::string text;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    float ttft_ms = 0.0F;
    float prefill_tokens_per_second = 0.0F;
    float decode_tokens_per_second = 0.0F;
};

class Qwen3Runner {
public:
    Qwen3Runner();
    ~Qwen3Runner();

    Qwen3Runner(const Qwen3Runner&) = delete;
    Qwen3Runner& operator=(const Qwen3Runner&) = delete;

    void Load(const ModelLoadOptions& options);
    InferenceResult Infer(const std::string& text,
                          const GenerationOptions& options = {});
    void ResetContext();
    bool IsLoaded() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace qwen3
