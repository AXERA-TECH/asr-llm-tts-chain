#include "kws_runtime.h"

#include "ax_engine_api.h"
#include "ax_sys_api.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

namespace kws {
namespace {

constexpr int kInputRate = 48000;
constexpr int kRate = 16000;
constexpr int kInputHop = 480;
constexpr int kHop = 160;
constexpr int kWindow = 480;
constexpr int kFft = 512;
constexpr int kMagnitudeBins = 224;
constexpr int kMelBins = 80;
constexpr int kMfccBins = 40;
constexpr int kClasses = 8;
constexpr int kStateCount = 6;
constexpr int kWarmupFrames = 100;
constexpr int kAlignment = 128;
constexpr std::size_t kFrontendFloats =
    kWindow + kMagnitudeBins * kMelBins + kMelBins * kMfccBins + kMfccBins;
constexpr const char* kStateInputs[kStateCount] = {
    "state_in_0", "state_in_1", "state_in_2",
    "state_in_3", "state_in_4", "state_in_5"};
constexpr const char* kStateOutputs[kStateCount] = {
    "state_out_0", "state_out_1", "state_out_2",
    "state_out_3", "state_out_4", "state_out_5"};

class Engine {
 public:
  ~Engine() { Release(); }

  void Init(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read AXModel: " + path);
    input.seekg(0, std::ios::end);
    const auto size = input.tellg();
    if (size <= 0 || size > static_cast<std::streamoff>(UINT32_MAX))
      throw std::runtime_error("invalid AXModel size");
    input.seekg(0);
    std::vector<char> model(static_cast<std::size_t>(size));
    input.read(model.data(), size);
    if (!input) throw std::runtime_error("truncated AXModel");
    const AX_S32 status = AX_ENGINE_CreateHandle(
        &handle_, model.data(), static_cast<AX_U32>(model.size()));
    if (status != 0 || !handle_)
      throw std::runtime_error("AX_ENGINE_CreateHandle failed: " +
                               std::to_string(status));
    const auto little = static_cast<AX_ENGINE_NPU_SET_T>(0x2U);
    if (AX_ENGINE_SetAffinity(handle_, little) != 0)
      throw std::runtime_error("cannot select LITTLE affinity 0x2");
    AX_ENGINE_NPU_SET_T actual = 0;
    if (AX_ENGINE_GetAffinity(handle_, &actual) != 0 || actual != little)
      throw std::runtime_error("KWS affinity is not LITTLE 0x2");
    if (AX_ENGINE_CreateContext(handle_) != 0 ||
        AX_ENGINE_GetIOInfo(handle_, &info_) != 0 || !info_)
      throw std::runtime_error("cannot create KWS AX Engine context");
    io_.nInputSize = info_->nInputSize;
    io_.nOutputSize = info_->nOutputSize;
    io_.pInputs = new AX_ENGINE_IO_BUFFER_T[io_.nInputSize]{};
    io_.pOutputs = new AX_ENGINE_IO_BUFFER_T[io_.nOutputSize]{};
    AX_S8 input_tag[] = "KWS-INPUT";
    AX_S8 output_tag[] = "KWS-OUTPUT";
    for (AX_U32 i = 0; i < io_.nInputSize; ++i) {
      io_.pInputs[i].nSize = info_->pInputs[i].nSize;
      if (AX_SYS_MemAlloc(&io_.pInputs[i].phyAddr, &io_.pInputs[i].pVirAddr,
                          io_.pInputs[i].nSize, kAlignment, input_tag) != 0)
        throw std::runtime_error("cannot allocate KWS input");
      inputs_[info_->pInputs[i].pName] = static_cast<int>(i);
    }
    for (AX_U32 i = 0; i < io_.nOutputSize; ++i) {
      io_.pOutputs[i].nSize = info_->pOutputs[i].nSize;
      if (AX_SYS_MemAlloc(&io_.pOutputs[i].phyAddr, &io_.pOutputs[i].pVirAddr,
                          io_.pOutputs[i].nSize, kAlignment, output_tag) != 0)
        throw std::runtime_error("cannot allocate KWS output");
      outputs_[info_->pOutputs[i].pName] = static_cast<int>(i);
    }
    if (InputSize("features") != kMfccBins * sizeof(float) ||
        OutputSize("logits") != kClasses * sizeof(float))
      throw std::runtime_error("KWS model feature/logit ABI mismatch");
    for (int i = 0; i < kStateCount; ++i) {
      if (InputSize(kStateInputs[i]) != OutputSize(kStateOutputs[i]))
        throw std::runtime_error("KWS external state ABI mismatch");
    }
    ResetStates();
    std::cerr << "[kws] loaded " << path
              << " affinity=0x2 (NPU1/LITTLE single core)\n";
  }

  void ResetStates() {
    for (const char* name : kStateInputs) {
      const int index = Input(name);
      if (index < 0) throw std::runtime_error("missing KWS state input");
      std::memset(io_.pInputs[index].pVirAddr, 0, io_.pInputs[index].nSize);
    }
  }

  std::array<float, kClasses> Run(const float* feature) {
    SetInput("features", feature, kMfccBins * sizeof(float));
    for (AX_U32 i = 0; i < io_.nInputSize; ++i) {
      if (AX_SYS_MflushCache(io_.pInputs[i].phyAddr, io_.pInputs[i].pVirAddr,
                             io_.pInputs[i].nSize) != 0)
        throw std::runtime_error("KWS input cache flush failed");
    }
    const AX_S32 status = AX_ENGINE_RunSync(handle_, &io_);
    if (status != 0)
      throw std::runtime_error("KWS inference failed: " + std::to_string(status));
    for (AX_U32 i = 0; i < io_.nOutputSize; ++i) {
      if (AX_SYS_MinvalidateCache(io_.pOutputs[i].phyAddr,
                                  io_.pOutputs[i].pVirAddr,
                                  io_.pOutputs[i].nSize) != 0)
        throw std::runtime_error("KWS output cache invalidate failed");
    }
    for (int i = 0; i < kStateCount; ++i) {
      const int from = Output(kStateOutputs[i]);
      const int to = Input(kStateInputs[i]);
      std::memcpy(io_.pInputs[to].pVirAddr, io_.pOutputs[from].pVirAddr,
                  io_.pInputs[to].nSize);
    }
    std::array<float, kClasses> logits{};
    const int index = Output("logits");
    std::memcpy(logits.data(), io_.pOutputs[index].pVirAddr, sizeof(logits));
    return logits;
  }

 private:
  int Input(const std::string& name) const {
    auto item = inputs_.find(name);
    return item == inputs_.end() ? -1 : item->second;
  }
  int Output(const std::string& name) const {
    auto item = outputs_.find(name);
    return item == outputs_.end() ? -1 : item->second;
  }
  std::size_t InputSize(const std::string& name) const {
    const int i = Input(name);
    return i < 0 ? 0 : io_.pInputs[i].nSize;
  }
  std::size_t OutputSize(const std::string& name) const {
    const int i = Output(name);
    return i < 0 ? 0 : io_.pOutputs[i].nSize;
  }
  void SetInput(const std::string& name, const void* data, std::size_t size) {
    const int i = Input(name);
    if (i < 0 || io_.pInputs[i].nSize != size)
      throw std::runtime_error("invalid KWS input");
    std::memcpy(io_.pInputs[i].pVirAddr, data, size);
  }
  void Release() {
    for (AX_U32 i = 0; io_.pInputs && i < io_.nInputSize; ++i)
      if (io_.pInputs[i].pVirAddr)
        AX_SYS_MemFree(io_.pInputs[i].phyAddr, io_.pInputs[i].pVirAddr);
    for (AX_U32 i = 0; io_.pOutputs && i < io_.nOutputSize; ++i)
      if (io_.pOutputs[i].pVirAddr)
        AX_SYS_MemFree(io_.pOutputs[i].phyAddr, io_.pOutputs[i].pVirAddr);
    delete[] io_.pInputs;
    delete[] io_.pOutputs;
    io_ = {};
    if (handle_) AX_ENGINE_DestroyHandle(handle_);
    handle_ = nullptr;
  }
  AX_ENGINE_HANDLE handle_ = nullptr;
  AX_ENGINE_IO_INFO_T* info_ = nullptr;
  AX_ENGINE_IO_T io_{};
  std::unordered_map<std::string, int> inputs_;
  std::unordered_map<std::string, int> outputs_;
};

std::vector<float> ReadFrontend(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot read KWS frontend: " + path);
  std::vector<float> values(kFrontendFloats);
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!input || input.peek() != std::ifstream::traits_type::eof())
    throw std::runtime_error("KWS frontend coefficient size mismatch");
  return values;
}

std::vector<std::string> ReadLabels(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot read KWS labels: " + path);
  std::vector<std::string> labels;
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (!line.empty()) labels.push_back(line);
  }
  if (labels.size() != kClasses)
    throw std::runtime_error("KWS labels must contain 8 classes");
  if (labels[0] != "_silence_" || labels[1] != "_unknown_")
    throw std::runtime_error("KWS labels 0/1 must be silence/unknown");
  return labels;
}

void Fft(std::array<std::complex<float>, kFft>* values) {
  auto& a = *values;
  for (int i = 1, j = 0; i < kFft; ++i) {
    int bit = kFft >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }
  constexpr float kPi = 3.14159265358979323846f;
  for (int length = 2; length <= kFft; length <<= 1) {
    const float angle = -2.0f * kPi / length;
    const std::complex<float> root(std::cos(angle), std::sin(angle));
    for (int i = 0; i < kFft; i += length) {
      std::complex<float> factor(1.0f, 0.0f);
      for (int j = 0; j < length / 2; ++j) {
        const auto even = a[i + j];
        const auto odd = a[i + j + length / 2] * factor;
        a[i + j] = even + odd;
        a[i + j + length / 2] = even - odd;
        factor *= root;
      }
    }
  }
}

}  // namespace

struct Runtime::Impl {
  explicit Impl(const apm_example::AppConfig::Kws& c,
                std::function<void()> lock_fn,
                std::function<void()> unlock_fn)
      : config(c), lock(std::move(lock_fn)), unlock(std::move(unlock_fn)),
        coefficients(ReadFrontend(c.frontend_path)), labels(ReadLabels(c.labels_path)),
        started(std::chrono::steady_clock::now()) {
    if (AX_SYS_Init() != 0)
      throw std::runtime_error("KWS AX_SYS_Init failed");
    AX_ENGINE_NPU_ATTR_T attributes{};
    attributes.eHardMode = AX_ENGINE_VIRTUAL_NPU_BIG_LITTLE;
    const AX_S32 init_status = AX_ENGINE_Init(&attributes);
    if (init_status != 0)
      throw std::runtime_error("KWS AX_ENGINE_Init failed: " +
                               std::to_string(init_status));
    AX_ENGINE_NPU_ATTR_T active{};
    if (AX_ENGINE_GetVNPUAttr(&active) != 0 ||
        active.eHardMode != AX_ENGINE_VIRTUAL_NPU_BIG_LITTLE)
      throw std::runtime_error("KWS requires BIG_LITTLE (2+1) VNPU mode");
    window = coefficients.data();
    mel = window + kWindow;
    dct = mel + kMagnitudeBins * kMelBins;
    normalization = dct + kMelBins * kMfccBins;
    BuildDecimator();
    engine.Init(c.model_path);
    Reset();
  }

  void BuildDecimator() {
    constexpr int taps = 63;
    constexpr double pi = 3.14159265358979323846;
    const double fc = 7200.0 / kInputRate;
    resample_coefficients.resize(taps);
    for (int i = 0; i < taps; ++i) {
      const double x = i - (taps - 1) / 2.0;
      const double sinc = x == 0 ? 2 * fc
          : std::sin(2 * pi * fc * x) / (pi * x);
      const double w = 0.42 - 0.5 * std::cos(2 * pi * i / (taps - 1)) +
                       0.08 * std::cos(4 * pi * i / (taps - 1));
      resample_coefficients[i] = sinc * w;
    }
    double gain = 0.0;
    for (double x : resample_coefficients) gain += x;
    for (double& x : resample_coefficients) x /= gain;
    resample_history.assign(taps - 1, 0.0f);
  }

  std::vector<std::int16_t> Resample(const std::vector<std::int16_t>& input) {
    std::vector<float> values = resample_history;
    values.reserve(values.size() + input.size());
    for (auto x : input) values.push_back(static_cast<float>(x));
    std::vector<std::int16_t> output;
    const std::size_t limit = values.size() - resample_history.size();
    for (std::size_t i = resample_phase; i < limit; i += 3) {
      double sum = 0.0;
      for (std::size_t j = 0; j < resample_coefficients.size(); ++j)
        sum += values[i + j] * resample_coefficients[j];
      output.push_back(static_cast<std::int16_t>(
          std::max(-32768.0, std::min(32767.0, std::round(sum)))));
    }
    resample_history.assign(values.end() - resample_history.size(), values.end());
    const std::size_t consumed = resample_phase < limit
        ? resample_phase + ((limit - resample_phase + 2) / 3) * 3
        : resample_phase;
    resample_phase = consumed >= limit ? consumed - limit : 0;
    return output;
  }

  std::array<float, kMfccBins> Feature(const std::array<float, kWindow>& audio) {
    std::array<std::complex<float>, kFft> spectrum{};
    for (int i = 0; i < kWindow; ++i)
      spectrum[i] = std::complex<float>(audio[i] * window[i], 0.0f);
    Fft(&spectrum);
    std::array<float, kMagnitudeBins> magnitude{};
    for (int i = 0; i < kMagnitudeBins; ++i)
      magnitude[i] = std::abs(spectrum[i]);
    std::array<float, kMelBins> log_mel{};
    for (int m = 0; m < kMelBins; ++m) {
      float value = 0.0f;
      for (int k = 0; k < kMagnitudeBins; ++k)
        value += magnitude[k] * mel[k * kMelBins + m];
      log_mel[m] = std::log(std::max(value, 1e-12f));
    }
    std::array<float, kMfccBins> feature{};
    for (int c = 0; c < kMfccBins; ++c) {
      float value = 0.0f;
      for (int m = 0; m < kMelBins; ++m)
        value += log_mel[m] * dct[m * kMfccBins + c];
      feature[c] = value * normalization[c];
    }
    return feature;
  }

  std::vector<Detection> Process(const std::vector<std::int16_t>& samples) {
    std::vector<Detection> detections;
    auto down = Resample(samples);
    pending.insert(pending.end(), down.begin(), down.end());
    while (pending.size() >= kHop) {
      std::array<float, kWindow> audio{};
      std::copy(history.begin(), history.end(), audio.begin());
      for (int i = 0; i < kHop; ++i)
        audio[history.size() + i] = pending[i] / 32768.0f;
      pending.erase(pending.begin(), pending.begin() + kHop);
      std::copy(audio.begin() + kHop, audio.end(), history.begin());
      const auto feature = Feature(audio);
      const auto begin = std::chrono::steady_clock::now();
      lock();
      std::array<float, kClasses> logits;
      try {
        logits = engine.Run(feature.data());
      } catch (...) {
        unlock();
        throw;
      }
      unlock();
      const auto elapsed = std::chrono::steady_clock::now() - begin;
      total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
      ++calls;
      ++frame;

      const int top = static_cast<int>(
          std::max_element(logits.begin(), logits.end()) - logits.begin());
      const float maximum = logits[top];
      float sum = 0.0f;
      for (float value : logits) sum += std::exp(value - maximum);
      const float probability = 1.0f / sum;
      if (processed < kWarmupFrames) ++processed;
      if (top >= 2 &&
          (probability < config.threshold ||
           frame < cooldown_until || top == last_reported)) {
        candidate = -1;
        stable = 0;
        continue;
      }
      if (candidate == top) {
        stable = std::min(stable + 1, config.stable_frames);
      } else {
        candidate = top;
        stable = 1;
      }
      if (processed >= kWarmupFrames && stable >= config.stable_frames &&
          candidate != last_reported) {
        last_reported = candidate;
        if (candidate >= 2) {
          Detection event{labels[candidate], probability, frame};
          detections.push_back(std::move(event));
          cooldown_until = frame +
              std::max<std::uint64_t>(1, (config.cooldown_ms + 9) / 10);
          candidate = -1;
          stable = 0;
        }
      }
    }
    return detections;
  }

  void Reset() {
    engine.ResetStates();
    pending.clear();
    history.fill(0.0f);
    resample_history.assign(resample_coefficients.size() - 1, 0.0f);
    resample_phase = 0;
    processed = stable = 0;
    candidate = -1;
  }

  apm_example::AppConfig::Kws config;
  std::function<void()> lock;
  std::function<void()> unlock;
  Engine engine;
  std::vector<float> coefficients;
  std::vector<std::string> labels;
  const float* window = nullptr;
  const float* mel = nullptr;
  const float* dct = nullptr;
  const float* normalization = nullptr;
  std::vector<double> resample_coefficients;
  std::vector<float> resample_history;
  std::size_t resample_phase = 0;
  std::deque<std::int16_t> pending;
  std::array<float, kWindow - kHop> history{};
  std::uint64_t frame = 0;
  std::uint64_t cooldown_until = 0;
  std::uint64_t calls = 0;
  std::uint64_t total_ns = 0;
  int processed = 0;
  int candidate = -1;
  int last_reported = -1;
  int stable = 0;
  std::chrono::steady_clock::time_point started;
};

Runtime::Runtime(const apm_example::AppConfig::Kws& config,
                 std::function<void()> lock, std::function<void()> unlock) {
  if (config.enabled)
    impl_ = std::make_unique<Impl>(config, std::move(lock), std::move(unlock));
}

Runtime::~Runtime() = default;
bool Runtime::enabled() const { return impl_ != nullptr; }

std::vector<Detection> Runtime::Process48k(
    const std::vector<std::int16_t>& samples) {
  return impl_ ? impl_->Process(samples) : std::vector<Detection>{};
}

void Runtime::Reset() {
  if (impl_) impl_->Reset();
}

void Runtime::Report() const {
  if (!impl_) return;
  const double total_ms = impl_->total_ns / 1e6;
  const double elapsed_ms = std::max(
      1.0, std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - impl_->started).count());
  std::cerr << "[npu] kws calls=" << impl_->calls
            << " avg=" << (impl_->calls ? total_ms / impl_->calls : 0.0)
            << "ms busy=" << total_ms << "ms util="
            << total_ms * 100.0 / elapsed_ms << "%\n";
}

}  // namespace kws
