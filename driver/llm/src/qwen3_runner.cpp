#include "qwen3_runner.hpp"

#include <ax_engine_api.h>
#include <ax_sys_api.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "runner/LLM.hpp"
#include "runner/LLMPostprocess.hpp"
#include "runner/utils/json.hpp"

namespace qwen3 {
namespace {

using Json = nlohmann::json;

std::string ResolvePath(const std::filesystem::path& base,
                        const std::string& value) {
    if (value.empty()) {
        return value;
    }
    const std::filesystem::path path(value);
    return path.is_absolute() ? path.string() : (base / path).string();
}

std::string StringValue(const Json& config, const char* key,
                        const std::string& fallback = {}) {
    const auto item = config.find(key);
    return item != config.end() && item->is_string()
               ? item->get<std::string>()
               : fallback;
}

template <typename T>
T Value(const Json& config, const char* key, const T& fallback) {
    const auto item = config.find(key);
    return item != config.end() && !item->is_null() ? item->get<T>() : fallback;
}

void ValidateGenerationOptions(const GenerationOptions& options) {
    if (options.max_tokens <= 0) {
        throw std::invalid_argument("max_tokens must be positive");
    }
    if (options.temperature < 0.0F) {
        throw std::invalid_argument("temperature must be >= 0");
    }
    if (options.top_p <= 0.0F || options.top_p > 1.0F) {
        throw std::invalid_argument("top_p must be in (0, 1]");
    }
    if (options.top_k < 0) {
        throw std::invalid_argument("top_k must be >= 0");
    }
    if (options.repetition_penalty <= 0.0F) {
        throw std::invalid_argument("repetition_penalty must be positive");
    }
}

::ThinkingMode ToAxThinkingMode(const qwen3::ThinkingMode mode) {
    switch (mode) {
        case qwen3::ThinkingMode::kEnabled:
            return ::ThinkingMode::Think;
        case qwen3::ThinkingMode::kDisabled:
            return ::ThinkingMode::NoThink;
        case qwen3::ThinkingMode::kDefault:
            return ::ThinkingMode::Unspecified;
    }
    return ::ThinkingMode::Unspecified;
}

}  // namespace

class Qwen3Runner::Impl {
public:
    ~Impl() { Unload(); }

    void Load(const ModelLoadOptions& options) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (loaded_) {
            throw std::logic_error("model is already loaded");
        }
        if (options.model_dir.empty()) {
            throw std::invalid_argument("model_dir is required");
        }

        const std::filesystem::path model_dir =
            std::filesystem::absolute(options.model_dir);
        const std::filesystem::path config_path = model_dir / "config.json";
        if (!std::filesystem::is_regular_file(config_path)) {
            throw std::runtime_error("config.json not found: " +
                                     config_path.string());
        }

        std::ifstream stream(config_path);
        if (!stream) {
            throw std::runtime_error("cannot open: " + config_path.string());
        }
        Json config;
        stream >> config;

        LLMAttrType attr;
        attr.system_prompt = options.system_prompt.empty()
                                 ? StringValue(config, "system_prompt",
                                               "You are a helpful assistant.")
                                 : options.system_prompt;
        attr.template_filename_axmodel = ResolvePath(
            model_dir, StringValue(config, "template_filename_axmodel"));
        attr.filename_post_axmodel = ResolvePath(
            model_dir, StringValue(config, "filename_post_axmodel"));
        attr.url_tokenizer_model = ResolvePath(
            model_dir, StringValue(config, "url_tokenizer_model"));
        attr.filename_tokens_embed = ResolvePath(
            model_dir, StringValue(config, "filename_tokens_embed"));
        attr.post_config_path = ResolvePath(
            model_dir,
            StringValue(config, "post_config_path", "post_config.json"));
        attr.tokenizer_type = StringValue(config, "tokenizer_type", "Qwen3");
        attr.axmodel_num = Value(config, "axmodel_num", 0);
        attr.tokens_embed_num = Value(config, "tokens_embed_num", 0);
        attr.tokens_embed_size = Value(config, "tokens_embed_size", 0);
        attr.pad_token_id = Value(config, "pad_token_id", 0);
        attr.hidden_size_per_layer_input =
            Value(config, "hidden_size_per_layer_input", 0);
        attr.rms_norm_eps = Value(config, "rms_norm_eps", 1.0e-6F);
        attr.b_use_mmap_load_embed = Value(
            config, "use_mmap_load_embed",
            Value(config, "b_use_mmap_load_embed", false));
#ifndef USE_AXCL
        attr.b_use_mmap_load_layer = Value(
            config, "use_mmap_load_layer",
            Value(config, "b_use_mmap_load_layer", true));
#endif
        attr.dynamic_load_enable = options.dynamic_load;
        attr.dynamic_load_pool_size =
            options.dynamic_load ? std::max(1, options.dynamic_load_pool_size) : 0;
        attr.mem_guard_enable = options.memory_guard;
        attr.mem_guard_floor_mb = options.memory_guard_floor_mb;
        attr.mem_guard_on_unsafe = "abort";

        if (attr.axmodel_num <= 0 || attr.tokens_embed_num <= 0 ||
            attr.tokens_embed_size <= 0 ||
            attr.template_filename_axmodel.empty() ||
            attr.filename_post_axmodel.empty() ||
            attr.url_tokenizer_model.empty() ||
            attr.filename_tokens_embed.empty()) {
            throw std::runtime_error("config.json is missing required model fields");
        }

        AX_ENGINE_NPU_ATTR_T npu_attr;
        std::memset(&npu_attr, 0, sizeof(npu_attr));
        // The audio pipeline keeps NPU1 available for realtime audio models;
        // Qwen3 was compiled for the BIG/NPU2 partition (device 0).
        npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_BIG_LITTLE;

        int status = AX_SYS_Init();
        if (status != 0) {
            throw std::runtime_error("AX_SYS_Init failed: " +
                                     std::to_string(status));
        }
        sys_initialized_ = true;

        status = AX_ENGINE_Init(&npu_attr);
        if (status != 0) {
            UnloadUnlocked();
            throw std::runtime_error("AX_ENGINE_Init failed: " +
                                     std::to_string(status));
        }
        engine_initialized_ = true;

        if (!llm_.Init(attr)) {
            const std::string error = llm_.GetLastError();
            UnloadUnlocked();
            throw std::runtime_error("LLM::Init failed" +
                                     (error.empty() ? std::string() : ": " + error));
        }

        model_options_ = options;
        model_options_.model_dir = model_dir.string();
        model_options_.system_prompt = attr.system_prompt;
        loaded_ = true;
    }

    InferenceResult Infer(const std::string& text,
                          const GenerationOptions& options) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!loaded_) {
            throw std::logic_error("model is not loaded");
        }
        if (text.empty()) {
            throw std::invalid_argument("input text must not be empty");
        }
        ValidateGenerationOptions(options);

        if (options.reset_context) {
            llm_.ResetKVCache();
            conversation_history_.clear();
        }

        auto* postprocess = llm_.getPostprocess();
        postprocess->set_repetition_penalty(
            options.repetition_penalty != 1.0F,
            options.repetition_penalty);
        postprocess->set_top_k_sampling(options.top_k > 0, options.top_k);

        const bool has_temperature = true;
        const bool has_top_p = options.top_p < 1.0F;
        llm_.SetRequestSamplingOverride(
            has_temperature, options.temperature, has_top_p, options.top_p,
            true, options.frequency_penalty, true, options.presence_penalty);
        llm_.SetRequestThinkingMode(ToAxThinkingMode(options.thinking));

        struct RequestStateGuard {
            explicit RequestStateGuard(LLM& llm) : llm(llm) {}
            ~RequestStateGuard() {
                llm.ClearRequestSamplingOverride();
                llm.ClearRequestThinkingMode();
                auto* post = llm.getPostprocess();
                post->set_repetition_penalty(false, 1.0F);
                post->set_top_k_sampling(false, 1);
                llm.getAttr()->runing_callback = nullptr;
                llm.getAttr()->reserve = nullptr;
            }
            LLM& llm;
        } request_guard(llm_);

        if (options.on_token) {
            llm_.getAttr()->runing_callback =
                [callback = options.on_token](std::string token, float, void*) {
                    callback(token);
                };
        } else {
            llm_.getAttr()->runing_callback =
                [](std::string, float, void*) {};
        }

        std::vector<Content> request_history = conversation_history_;
        if (request_history.empty()) {
            request_history.push_back(
                {SYSTEM, TEXT, model_options_.system_prompt, 0, 0});
        }
        request_history.push_back({USER, TEXT, text, 0, 0});

        const std::size_t input_history_size = request_history.size();
        const std::vector<Content> output =
            llm_.Run(std::move(request_history), options.max_tokens);
        if (output.size() <= input_history_size || output.back().role != ASSISTANT) {
            const std::string error = llm_.GetLastError();
            throw std::runtime_error("inference failed" +
                                     (error.empty() ? std::string() : ": " + error));
        }
        conversation_history_ = output;

        InferenceResult result;
        result.text = output.back().data;
        result.prompt_tokens = llm_.GetLastPromptTokenNum();
        result.completion_tokens = llm_.GetLastCompletionTokenNum();
        result.ttft_ms = llm_.GetLastTtftMs();
        result.prefill_tokens_per_second = llm_.GetLastPrefillTps();
        result.decode_tokens_per_second = llm_.GetLastDecodeTps();
        return result;
    }

    void ResetContext() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!loaded_) {
            throw std::logic_error("model is not loaded");
        }
        llm_.ResetKVCache();
        conversation_history_.clear();
    }

    bool IsLoaded() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return loaded_;
    }

private:
    void Unload() {
        std::lock_guard<std::mutex> lock(mutex_);
        UnloadUnlocked();
    }

    void UnloadUnlocked() {
        if (loaded_) {
            llm_.Deinit();
            loaded_ = false;
        }
        if (engine_initialized_) {
            AX_ENGINE_Deinit();
            engine_initialized_ = false;
        }
        if (sys_initialized_) {
            AX_SYS_Deinit();
            sys_initialized_ = false;
        }
    }

    mutable std::mutex mutex_;
    LLM llm_;
    std::vector<Content> conversation_history_;
    ModelLoadOptions model_options_;
    bool sys_initialized_ = false;
    bool engine_initialized_ = false;
    bool loaded_ = false;
};

Qwen3Runner::Qwen3Runner() : impl_(std::make_unique<Impl>()) {}
Qwen3Runner::~Qwen3Runner() = default;

void Qwen3Runner::Load(const ModelLoadOptions& options) {
    impl_->Load(options);
}

InferenceResult Qwen3Runner::Infer(const std::string& text,
                                   const GenerationOptions& options) {
    return impl_->Infer(text, options);
}

void Qwen3Runner::ResetContext() { impl_->ResetContext(); }
bool Qwen3Runner::IsLoaded() const noexcept { return impl_->IsLoaded(); }

}  // namespace qwen3
