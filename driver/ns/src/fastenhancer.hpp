#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class FastEnhancer {
public:
    static constexpr int kSampleRate = 48000;
    static constexpr int kNfft = 1024;
    static constexpr int kHop = 512;
    static constexpr int kCacheLength = kNfft - kHop;
    static constexpr int kFrequencyBins = kNfft / 2 + 1;
    static constexpr int kModelBins = kNfft / 2;
    static constexpr int kGruElements = 24 * 20;

    FastEnhancer();
    ~FastEnhancer();

    FastEnhancer(const FastEnhancer&) = delete;
    FastEnhancer& operator=(const FastEnhancer&) = delete;

    bool load(const std::string& model_path);
    bool process(const float* input, float* output);
    const std::string& last_error() const { return last_error_; }

private:
    void shutdown();
    bool run_core(const float* compressed_spectrum, float* mask);
    void stft(const float* input, float* spectrum);
    void istft(const float* spectrum, float* output);
    bool fail(const std::string& message);

    void* handle_ = nullptr;
    void* io_ = nullptr;
    bool system_initialized_ = false;
    bool engine_initialized_ = false;
    bool loaded_ = false;
    std::vector<std::uint8_t> model_data_;
    std::array<float, kNfft> window_{};
    std::array<float, kNfft> inverse_window_{};
    std::array<float, kCacheLength> stft_cache_{};
    std::array<float, kCacheLength> istft_cache_{};
    std::array<float, kGruElements> gru_cache_0_{};
    std::array<float, kGruElements> gru_cache_1_{};
    std::string last_error_;
};
