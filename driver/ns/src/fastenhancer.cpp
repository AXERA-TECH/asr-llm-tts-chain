#include "fastenhancer.hpp"

#include "ax_engine_api.h"
#include "ax_engine_type.h"
#include "ax_sys_api.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr float kCompressionPower = -0.7f;
constexpr float kDecompressionPower = 1.0f / 0.3f - 1.0f;

void fft(float* real, float* imaginary, int size, bool inverse) {
    for (int i = 1, j = 0; i < size; ++i) {
        int bit = size >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imaginary[i], imaginary[j]);
        }
    }

    for (int length = 2; length <= size; length <<= 1) {
        const double angle = (inverse ? 2.0 : -2.0) * kPi / length;
        const double step_real = std::cos(angle);
        const double step_imaginary = std::sin(angle);
        for (int base = 0; base < size; base += length) {
            double twiddle_real = 1.0;
            double twiddle_imaginary = 0.0;
            for (int offset = 0; offset < length / 2; ++offset) {
                const int even = base + offset;
                const int odd = even + length / 2;
                const float odd_real = static_cast<float>(
                    twiddle_real * real[odd] - twiddle_imaginary * imaginary[odd]);
                const float odd_imaginary = static_cast<float>(
                    twiddle_real * imaginary[odd] + twiddle_imaginary * real[odd]);
                real[odd] = real[even] - odd_real;
                imaginary[odd] = imaginary[even] - odd_imaginary;
                real[even] += odd_real;
                imaginary[even] += odd_imaginary;
                const double next_real =
                    twiddle_real * step_real - twiddle_imaginary * step_imaginary;
                twiddle_imaginary =
                    twiddle_real * step_imaginary + twiddle_imaginary * step_real;
                twiddle_real = next_real;
            }
        }
    }

    if (inverse) {
        for (int i = 0; i < size; ++i) {
            real[i] /= size;
            imaginary[i] /= size;
        }
    }
}

}  // namespace

FastEnhancer::FastEnhancer() {
    for (int i = 0; i < kNfft; ++i) {
        const float window = static_cast<float>(
            0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(i) / kNfft));
        window_[i] = window;
    }
    for (int i = 0; i < kNfft; ++i) {
        const float shifted = window_[(i + kHop) % kNfft];
        const float normalization = window_[i] * window_[i] + shifted * shifted;
        inverse_window_[i] = normalization > 0.0f ? window_[i] / normalization : 0.0f;
    }
}

FastEnhancer::~FastEnhancer() {
    shutdown();
}

bool FastEnhancer::fail(const std::string& message) {
    last_error_ = message;
    return false;
}

void FastEnhancer::shutdown() {
    if (io_ != nullptr) {
        auto* io = static_cast<AX_ENGINE_IO_T*>(io_);
        if (io->pInputs != nullptr) {
            for (AX_U32 i = 0; i < io->nInputSize; ++i) {
                if (io->pInputs[i].pVirAddr != nullptr) {
                    AX_SYS_MemFree(io->pInputs[i].phyAddr, io->pInputs[i].pVirAddr);
                }
            }
        }
        if (io->pOutputs != nullptr) {
            for (AX_U32 i = 0; i < io->nOutputSize; ++i) {
                if (io->pOutputs[i].pVirAddr != nullptr) {
                    AX_SYS_MemFree(io->pOutputs[i].phyAddr, io->pOutputs[i].pVirAddr);
                }
            }
        }
        delete[] io->pInputs;
        delete[] io->pOutputs;
        delete io;
        io_ = nullptr;
    }
    if (handle_ != nullptr) {
        AX_ENGINE_DestroyHandle(handle_);
        handle_ = nullptr;
    }
    if (engine_initialized_) {
        AX_ENGINE_Deinit();
        engine_initialized_ = false;
    }
    if (system_initialized_) {
        AX_SYS_Deinit();
        system_initialized_ = false;
    }
    loaded_ = false;
    model_data_.clear();
}

bool FastEnhancer::load(const std::string& model_path) {
    shutdown();
    last_error_.clear();

    std::ifstream input(model_path, std::ios::binary | std::ios::ate);
    if (!input) return fail("cannot open model: " + model_path);
    const std::streamoff end = input.tellg();
    if (end <= 0 || static_cast<unsigned long long>(end) >
                        std::numeric_limits<AX_U32>::max()) {
        return fail("invalid model size: " + model_path);
    }
    model_data_.resize(static_cast<std::size_t>(end));
    input.seekg(0);
    if (!input.read(reinterpret_cast<char*>(model_data_.data()), end)) {
        return fail("cannot read model: " + model_path);
    }

    AX_S32 status = AX_SYS_Init();
    if (status != 0) return fail("AX_SYS_Init failed: " + std::to_string(status));
    system_initialized_ = true;

    AX_ENGINE_NPU_ATTR_T attributes{};
    // BIG_LITTLE exposes affinity 0x1 as the two-core BIG partition for
    // SenseVoice uses affinity 0x1; FastEnhance uses the single-core LITTLE
    // realtime partition at affinity 0x2.
    attributes.eHardMode = AX_ENGINE_VIRTUAL_NPU_BIG_LITTLE;
    status = AX_ENGINE_Init(&attributes);
    if (status != 0) {
        fail("AX_ENGINE_Init failed: " + std::to_string(status));
        shutdown();
        return false;
    }
    engine_initialized_ = true;

    status = AX_ENGINE_CreateHandle(&handle_, model_data_.data(),
                                    static_cast<AX_U32>(model_data_.size()));
    if (status != 0 || handle_ == nullptr) {
        fail("AX_ENGINE_CreateHandle failed: " + std::to_string(status));
        shutdown();
        return false;
    }
    // FastEnhance is an NPU1 model and runs on the LITTLE one-core partition.
    AX_ENGINE_NPU_ATTR_T active_attributes{};
    status = AX_ENGINE_GetVNPUAttr(&active_attributes);
    if (status != 0) {
        fail("AX_ENGINE_GetVNPUAttr failed: " + std::to_string(status));
        shutdown();
        return false;
    }
    if (active_attributes.eHardMode != AX_ENGINE_VIRTUAL_NPU_BIG_LITTLE) {
        fail("FastEnhance requires AX_ENGINE_VIRTUAL_NPU_BIG_LITTLE");
        shutdown();
        return false;
    }
    const AX_ENGINE_NPU_SET_T single_vnpu =
        static_cast<AX_ENGINE_NPU_SET_T>(0x2U);
    if (AX_ENGINE_SetAffinity(handle_, single_vnpu) != 0) {
        AX_ENGINE_NPU_SET_T actual = 0;
        if (AX_ENGINE_GetAffinity(handle_, &actual) != 0 ||
            actual != single_vnpu) {
            fail("AX_ENGINE_SetAffinity(VNPU1 single-core) failed");
            shutdown();
            return false;
        }
    }
    status = AX_ENGINE_CreateContext(handle_);
    if (status != 0) {
        fail("AX_ENGINE_CreateContext failed: " + std::to_string(status));
        shutdown();
        return false;
    }

    AX_ENGINE_IO_INFO_T* info = nullptr;
    status = AX_ENGINE_GetIOInfo(handle_, &info);
    if (status != 0 || info == nullptr || info->nInputSize != 3 || info->nOutputSize != 3) {
        fail("unexpected model I/O layout");
        shutdown();
        return false;
    }
    constexpr AX_U32 spectrum_bytes = kModelBins * 2 * sizeof(float);
    constexpr AX_U32 cache_bytes = kGruElements * sizeof(float);
    if (info->pInputs[0].nSize < spectrum_bytes || info->pOutputs[0].nSize < spectrum_bytes ||
        info->pInputs[1].nSize < cache_bytes || info->pInputs[2].nSize < cache_bytes ||
        info->pOutputs[1].nSize < cache_bytes || info->pOutputs[2].nSize < cache_bytes) {
        fail("model I/O buffers do not match the 48 kHz FastEnhancer model");
        shutdown();
        return false;
    }

    auto* io = new (std::nothrow) AX_ENGINE_IO_T{};
    if (io == nullptr) {
        fail("cannot allocate AX_ENGINE I/O descriptor");
        shutdown();
        return false;
    }
    io_ = io;
    io->nInputSize = info->nInputSize;
    io->nOutputSize = info->nOutputSize;
    io->pInputs = new (std::nothrow) AX_ENGINE_IO_BUFFER_T[io->nInputSize]{};
    io->pOutputs = new (std::nothrow) AX_ENGINE_IO_BUFFER_T[io->nOutputSize]{};
    if (io->pInputs == nullptr || io->pOutputs == nullptr) {
        fail("cannot allocate AX_ENGINE buffer descriptors");
        shutdown();
        return false;
    }

    for (AX_U32 i = 0; i < io->nInputSize; ++i) {
        io->pInputs[i].nSize = info->pInputs[i].nSize;
        status = AX_SYS_MemAlloc(&io->pInputs[i].phyAddr, &io->pInputs[i].pVirAddr,
                                 io->pInputs[i].nSize, 128,
                                 reinterpret_cast<const AX_S8*>("fastenhancer"));
        if (status != 0) {
            fail("AX_SYS_MemAlloc input failed: " + std::to_string(status));
            shutdown();
            return false;
        }
        std::memset(io->pInputs[i].pVirAddr, 0, io->pInputs[i].nSize);
    }
    for (AX_U32 i = 0; i < io->nOutputSize; ++i) {
        io->pOutputs[i].nSize = info->pOutputs[i].nSize;
        status = AX_SYS_MemAlloc(&io->pOutputs[i].phyAddr, &io->pOutputs[i].pVirAddr,
                                 io->pOutputs[i].nSize, 128,
                                 reinterpret_cast<const AX_S8*>("fastenhancer"));
        if (status != 0) {
            fail("AX_SYS_MemAlloc output failed: " + std::to_string(status));
            shutdown();
            return false;
        }
    }

    stft_cache_.fill(0.0f);
    istft_cache_.fill(0.0f);
    gru_cache_0_.fill(0.0f);
    gru_cache_1_.fill(0.0f);
    loaded_ = true;
    return true;
}

void FastEnhancer::stft(const float* input, float* spectrum) {
    std::array<float, kNfft> real{};
    std::array<float, kNfft> imaginary{};
    std::copy(stft_cache_.begin(), stft_cache_.end(), real.begin());
    std::copy_n(input, kHop, real.begin() + kCacheLength);
    std::copy_n(input, kHop, stft_cache_.begin());
    for (int i = 0; i < kNfft; ++i) real[i] *= window_[i];
    fft(real.data(), imaginary.data(), kNfft, false);
    for (int bin = 0; bin < kFrequencyBins; ++bin) {
        spectrum[bin * 2] = real[bin];
        spectrum[bin * 2 + 1] = imaginary[bin];
    }
}

void FastEnhancer::istft(const float* spectrum, float* output) {
    std::array<float, kNfft> real{};
    std::array<float, kNfft> imaginary{};
    for (int bin = 0; bin < kFrequencyBins; ++bin) {
        real[bin] = spectrum[bin * 2];
        imaginary[bin] = spectrum[bin * 2 + 1];
    }
    for (int bin = 1; bin < kNfft / 2; ++bin) {
        real[kNfft - bin] = real[bin];
        imaginary[kNfft - bin] = -imaginary[bin];
    }
    fft(real.data(), imaginary.data(), kNfft, true);
    for (int i = 0; i < kNfft; ++i) real[i] *= inverse_window_[i];
    for (int i = 0; i < kCacheLength; ++i) real[i] += istft_cache_[i];
    std::copy_n(real.begin(), kHop, output);
    std::copy_n(real.begin() + kHop, kCacheLength, istft_cache_.begin());
}

bool FastEnhancer::run_core(const float* compressed_spectrum, float* mask) {
    constexpr std::size_t spectrum_bytes = kModelBins * 2 * sizeof(float);
    constexpr std::size_t cache_bytes = kGruElements * sizeof(float);
    auto* io = static_cast<AX_ENGINE_IO_T*>(io_);
    std::memcpy(io->pInputs[0].pVirAddr, compressed_spectrum, spectrum_bytes);
    std::memcpy(io->pInputs[1].pVirAddr, gru_cache_0_.data(), cache_bytes);
    std::memcpy(io->pInputs[2].pVirAddr, gru_cache_1_.data(), cache_bytes);
    const AX_S32 status = AX_ENGINE_RunSync(handle_, io);
    if (status != 0) return fail("AX_ENGINE_RunSync failed: " + std::to_string(status));
    std::memcpy(mask, io->pOutputs[0].pVirAddr, spectrum_bytes);
    std::memcpy(gru_cache_0_.data(), io->pOutputs[1].pVirAddr, cache_bytes);
    std::memcpy(gru_cache_1_.data(), io->pOutputs[2].pVirAddr, cache_bytes);
    return true;
}

bool FastEnhancer::process(const float* input, float* output) {
    if (!loaded_) return fail("model is not loaded");

    std::array<float, kFrequencyBins * 2> spectrum{};
    std::array<float, kModelBins * 2> compressed{};
    std::array<float, kModelBins * 2> mask{};
    std::array<float, kFrequencyBins * 2> enhanced{};
    stft(input, spectrum.data());

    for (int bin = 0; bin < kModelBins; ++bin) {
        const float real = spectrum[bin * 2];
        const float imaginary = spectrum[bin * 2 + 1];
        const float magnitude = std::max(std::hypot(real, imaginary), 1.0e-5f);
        const float scale = std::pow(magnitude, kCompressionPower);
        compressed[bin * 2] = real * scale;
        compressed[bin * 2 + 1] = imaginary * scale;
    }
    if (!run_core(compressed.data(), mask.data())) return false;

    for (int bin = 0; bin < kModelBins; ++bin) {
        const float real = compressed[bin * 2];
        const float imaginary = compressed[bin * 2 + 1];
        const float mask_real = mask[bin * 2];
        const float mask_imaginary = mask[bin * 2 + 1];
        const float masked_real = real * mask_real - imaginary * mask_imaginary;
        const float masked_imaginary = real * mask_imaginary + imaginary * mask_real;
        const float magnitude = std::hypot(masked_real, masked_imaginary);
        const float scale = std::pow(magnitude, kDecompressionPower);
        enhanced[bin * 2] = masked_real * scale;
        enhanced[bin * 2 + 1] = masked_imaginary * scale;
    }
    enhanced[kModelBins * 2] = 0.0f;
    enhanced[kModelBins * 2 + 1] = 0.0f;
    istft(enhanced.data(), output);
    for (int i = 0; i < kHop; ++i) output[i] = std::clamp(output[i], -1.0f, 1.0f);
    return true;
}
