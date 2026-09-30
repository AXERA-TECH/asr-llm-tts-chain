#include "alsa-apm-backend.h"

#include <alsa/asoundlib.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "api/scoped_refptr.h"
#include "common_audio/resampler/push_sinc_resampler.h"
#include "modules/audio_processing/include/audio_processing.h"

namespace apm_example {
namespace {

constexpr int kApmFrameMs = 10;

class SampleQueue {
 public:
  explicit SampleQueue(size_t capacity) : capacity_(capacity) {}

  size_t Write(const int16_t* input,
               size_t count,
               std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    size_t written = 0;
    std::unique_lock<std::mutex> lock(mutex_);
    while (written < count && !closed_) {
      if (samples_.size() == capacity_) {
        if (!not_full_.wait_until(lock, deadline, [&] {
              return samples_.size() < capacity_ || closed_;
            })) {
          break;
        }
        continue;
      }
      const size_t n = std::min(count - written, capacity_ - samples_.size());
      samples_.insert(samples_.end(), input + written, input + written + n);
      written += n;
      not_empty_.notify_one();
    }
    return written;
  }

  size_t Read(int16_t* output,
              size_t count,
              std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    size_t read = 0;
    std::unique_lock<std::mutex> lock(mutex_);
    while (read < count) {
      if (samples_.empty()) {
        if (closed_ || !not_empty_.wait_until(lock, deadline, [&] {
              return !samples_.empty() || closed_;
            })) {
          break;
        }
        continue;
      }
      const size_t n = std::min(count - read, samples_.size());
      for (size_t i = 0; i < n; ++i) {
        output[read + i] = samples_.front();
        samples_.pop_front();
      }
      read += n;
      not_full_.notify_one();
    }
    return read;
  }

  // Used only by the real-time capture path: never wait for a slow consumer.
  size_t WriteDroppingOldest(const int16_t* input, size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
      return count;
    }
    size_t dropped = 0;
    if (count >= capacity_) {
      dropped = samples_.size() + count - capacity_;
      samples_.clear();
      input += count - capacity_;
      count = capacity_;
    } else if (samples_.size() + count > capacity_) {
      dropped = samples_.size() + count - capacity_;
      for (size_t i = 0; i < dropped; ++i) {
        samples_.pop_front();
      }
    }
    samples_.insert(samples_.end(), input, input + count);
    not_empty_.notify_one();
    return dropped;
  }

  // Playback must not wait: unavailable input is filled with silence.
  size_t ReadAvailable(int16_t* output, size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t n = std::min(count, samples_.size());
    for (size_t i = 0; i < n; ++i) {
      output[i] = samples_.front();
      samples_.pop_front();
    }
    not_full_.notify_one();
    return n;
  }

  void Close() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    not_empty_.notify_all();
    not_full_.notify_all();
  }

 private:
  const size_t capacity_;
  std::deque<int16_t> samples_;
  bool closed_ = false;
  std::mutex mutex_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
};

std::string AlsaError(const char* operation, int error) {
  return std::string(operation) + ": " + snd_strerror(error);
}

uint16_t ReadLe16(const unsigned char* value) {
  return static_cast<uint16_t>(value[0]) |
         (static_cast<uint16_t>(value[1]) << 8);
}

uint32_t ReadLe32(const unsigned char* value) {
  return static_cast<uint32_t>(value[0]) |
         (static_cast<uint32_t>(value[1]) << 8) |
         (static_cast<uint32_t>(value[2]) << 16) |
         (static_cast<uint32_t>(value[3]) << 24);
}

}  // namespace

class AlsaApmBackend::Impl {
 public:
  explicit Impl(AppConfig config)
      : config_(std::move(config)),
        frames_per_apm_frame_(config_.audio.capture_sample_rate_hz *
                              kApmFrameMs / 1000),
        playback_samples_per_apm_frame_(frames_per_apm_frame_ *
                                        config_.audio.playback_channels),
        playback_queue_(
            static_cast<size_t>(config_.audio.playback_sample_rate_hz) *
            config_.audio.playback_channels *
            config_.buffers.application_buffer_ms / 1000),
        recorded_queue_(
            static_cast<size_t>(config_.audio.capture_sample_rate_hz) *
            config_.buffers.application_buffer_ms / 1000) {}

  ~Impl() { Stop(); }

  bool Start() {
    if (running_.load()) {
      return true;
    }
    std::string validation_error;
    if (!ValidateApmConfig(config_, &validation_error)) {
      SetError(validation_error);
      return false;
    }

    const bool file_input = config_.runtime.input == "file";
    if (file_input && !OpenWaveInput()) {
      return false;
    }

    int error = 0;
    if (config_.runtime.playback_enabled) {
      error = snd_pcm_open(&playback_pcm_,
                           config_.audio.playback_device.c_str(),
                           SND_PCM_STREAM_PLAYBACK, 0);
      if (error < 0) {
        SetError(AlsaError("open playback device", error));
        CloseDevices();
        return false;
      }
    }
    if (!file_input) {
      error = snd_pcm_open(&capture_pcm_, config_.audio.capture_device.c_str(),
                           SND_PCM_STREAM_CAPTURE, 0);
      if (error < 0) {
        SetError(AlsaError("open capture device", error));
        CloseDevices();
        return false;
      }
    }

    if ((config_.runtime.playback_enabled &&
         !ConfigurePcm(playback_pcm_, true,
                       config_.audio.playback_sample_rate_hz,
                       config_.audio.playback_channels,
                       &playback_period_frames_, nullptr)) ||
        (!file_input && !ConfigurePcm(capture_pcm_, false,
                                      config_.audio.capture_sample_rate_hz,
                                      config_.audio.capture_channels,
                                      &capture_period_frames_,
                                      &capture_channels_))) {
      CloseDevices();
      return false;
    }

    apm_ = webrtc::AudioProcessingBuilder().Create();
    if (!apm_) {
      SetError("AudioProcessingBuilder::Create() failed");
      CloseDevices();
      return false;
    }
    webrtc::AudioProcessing::Config config;
    config.echo_canceller.enabled = config_.apm.aec.enabled;
    config.echo_canceller.mobile_mode = config_.apm.aec.mode == "mobile";
    // WebRTC's multi-channel AEC paths are meaningful only for an actual
    // multi-channel stream.  In particular, keep mono capture on the normal
    // one-channel path even when the shared config was authored for stereo.
    config.pipeline.multi_channel_render =
        config_.apm.aec.multi_channel_render &&
        config_.audio.playback_channels > 1;
    config.pipeline.multi_channel_capture =
        config_.apm.aec.multi_channel_capture && capture_channels_ > 1;
    config.noise_suppression.enabled = config_.apm.ns.enabled;
    using Ns = webrtc::AudioProcessing::Config::NoiseSuppression;
    if (config_.apm.ns.level == "low") {
      config.noise_suppression.level = Ns::kLow;
    } else if (config_.apm.ns.level == "moderate") {
      config.noise_suppression.level = Ns::kModerate;
    } else if (config_.apm.ns.level == "high") {
      config.noise_suppression.level = Ns::kHigh;
    } else {
      config.noise_suppression.level = Ns::kVeryHigh;
    }
    config.high_pass_filter.enabled = config_.apm.high_pass_filter_enabled;

    config.gain_controller1.enabled = config_.apm.agc1.enabled;
    using Agc1 = webrtc::AudioProcessing::Config::GainController1;
    if (config_.apm.agc1.mode == "adaptive_analog") {
      config.gain_controller1.mode = Agc1::kAdaptiveAnalog;
    } else if (config_.apm.agc1.mode == "adaptive_digital") {
      config.gain_controller1.mode = Agc1::kAdaptiveDigital;
    } else {
      config.gain_controller1.mode = Agc1::kFixedDigital;
    }
    config.gain_controller1.target_level_dbfs =
        config_.apm.agc1.target_level_dbfs;
    config.gain_controller1.compression_gain_db =
        config_.apm.agc1.compression_gain_db;
    config.gain_controller1.enable_limiter = config_.apm.agc1.limiter;

    config.gain_controller2.enabled = config_.apm.agc2.enabled;
    config.gain_controller2.fixed_digital.gain_db =
        config_.apm.agc2.fixed_gain_db;
    config.gain_controller2.adaptive_digital.enabled =
        config_.apm.agc2.adaptive_digital;
    config.gain_controller2.adaptive_digital.headroom_db =
        config_.apm.agc2.adaptive_headroom_db;
    config.gain_controller2.adaptive_digital.max_gain_db =
        config_.apm.agc2.adaptive_max_gain_db;
    config.gain_controller2.adaptive_digital.initial_gain_db =
        config_.apm.agc2.adaptive_initial_gain_db;
    config.gain_controller2.adaptive_digital.max_gain_change_db_per_second =
        config_.apm.agc2.adaptive_max_gain_change_db_per_second;
    config.gain_controller2.adaptive_digital.max_output_noise_level_dbfs =
        config_.apm.agc2.adaptive_max_output_noise_level_dbfs;
    config.capture_level_adjustment.enabled =
        config_.apm.gain_adjustment.enabled;
    config.capture_level_adjustment.pre_gain_factor =
        config_.apm.gain_adjustment.pre_gain_factor;
    config.capture_level_adjustment.post_gain_factor =
        config_.apm.gain_adjustment.post_gain_factor;
    apm_->ApplyConfig(config);

    running_.store(true);
    if (config_.runtime.playback_enabled) {
      playback_thread_ = std::thread(&Impl::PlaybackLoop, this);
    }
    capture_thread_ = file_input
                          ? std::thread(&Impl::FileCaptureLoop, this)
                          : std::thread(&Impl::CaptureLoop, this);
    return true;
  }

  void Stop() {
    running_.store(false);
    playback_queue_.Close();
    recorded_queue_.Close();

    // Drop/abort wakes blocking snd_pcm_readi()/snd_pcm_writei().
    if (capture_pcm_) {
      snd_pcm_drop(capture_pcm_);
    }
    if (playback_pcm_) {
      snd_pcm_drop(playback_pcm_);
    }
    if (capture_thread_.joinable()) {
      capture_thread_.join();
    }
    if (playback_thread_.joinable()) {
      playback_thread_.join();
    }
    CloseDevices();
    if (file_input_.is_open()) {
      file_input_.close();
    }
    apm_ = nullptr;
  }

  size_t WritePlayback(const int16_t* samples,
                       size_t count,
                       std::chrono::milliseconds timeout) {
    if (!samples || !running_.load()) {
      return 0;
    }
    return playback_queue_.Write(samples, count, timeout);
  }

  size_t ReadRecorded(int16_t* samples,
                      size_t count,
                      std::chrono::milliseconds timeout) {
    if (!samples) {
      return 0;
    }
    return recorded_queue_.Read(samples, count, timeout);
  }

  bool input_finished() const { return input_finished_.load(); }

  std::string last_error() const {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return last_error_;
  }

  uint64_t playback_underrun_samples() const {
    return playback_underrun_samples_.load();
  }

  uint64_t dropped_recorded_samples() const {
    return dropped_recorded_samples_.load();
  }

  bool OpenWaveInput() {
    file_input_.open(config_.runtime.input_wav, std::ios::binary);
    if (!file_input_) {
      SetError("input WAV does not exist or cannot be opened: " +
               config_.runtime.input_wav);
      return false;
    }

    unsigned char riff[12] = {};
    if (!file_input_.read(reinterpret_cast<char*>(riff), sizeof(riff)) ||
        std::memcmp(riff, "RIFF", 4) != 0 ||
        std::memcmp(riff + 8, "WAVE", 4) != 0) {
      SetError("file input is not a RIFF/WAVE file: " +
               config_.runtime.input_wav);
      return false;
    }

    bool have_format = false;
    bool have_data = false;
    uint16_t audio_format = 0;
    uint16_t channels = 0;
    uint16_t block_align = 0;
    uint16_t bits_per_sample = 0;
    uint32_t sample_rate = 0;
    std::streampos data_position{};
    uint32_t data_size = 0;
    while (file_input_) {
      unsigned char header[8] = {};
      if (!file_input_.read(reinterpret_cast<char*>(header), sizeof(header))) {
        break;
      }
      const uint32_t chunk_size = ReadLe32(header + 4);
      const std::streampos payload = file_input_.tellg();
      if (std::memcmp(header, "fmt ", 4) == 0) {
        if (chunk_size < 16) {
          SetError("input WAV has an invalid fmt chunk");
          return false;
        }
        unsigned char format[16] = {};
        if (!file_input_.read(reinterpret_cast<char*>(format), sizeof(format))) {
          SetError("input WAV fmt chunk is truncated");
          return false;
        }
        audio_format = ReadLe16(format);
        channels = ReadLe16(format + 2);
        sample_rate = ReadLe32(format + 4);
        block_align = ReadLe16(format + 12);
        bits_per_sample = ReadLe16(format + 14);
        have_format = true;
      } else if (std::memcmp(header, "data", 4) == 0) {
        data_position = payload;
        data_size = chunk_size;
        have_data = true;
      }
      if (have_format && have_data) {
        break;
      }
      file_input_.seekg(payload +
                        static_cast<std::streamoff>(chunk_size +
                                                    (chunk_size & 1u)));
    }

    if (!have_format || !have_data) {
      SetError("input WAV is missing a fmt or data chunk");
      return false;
    }
    if (audio_format != 1 || bits_per_sample != 16 || channels == 0 ||
        channels > 8 || block_align != channels * sizeof(int16_t) ||
        sample_rate < 1000 || sample_rate > 384000 || data_size == 0 ||
        data_size % block_align != 0) {
      SetError("input WAV must be non-empty PCM S16_LE with 1-8 channels "
               "and a 1-384 kHz sample rate");
      return false;
    }
    file_input_.clear();
    file_input_.seekg(data_position);
    if (!file_input_) {
      SetError("cannot seek to input WAV audio data");
      return false;
    }
    capture_channels_ = channels;
    file_sample_rate_hz_ = sample_rate;
    file_bytes_remaining_ = data_size;
    if (file_sample_rate_hz_ !=
        static_cast<uint32_t>(config_.audio.capture_sample_rate_hz)) {
      std::cerr << "Resampling file input from " << file_sample_rate_hz_
                << " Hz to " << config_.audio.capture_sample_rate_hz
                << " Hz with the WebRTC sinc resampler\n";
    }
    return true;
  }

  bool ProcessCaptureFrame(int16_t* apm_frame,
                           int capture_channels,
                           int capture_delay_ms,
                           std::vector<int16_t>* processed_left) {
    const webrtc::StreamConfig stream_config(
        config_.audio.capture_sample_rate_hz, capture_channels);
    if (config_.apm.aec.enabled) {
      const int total_delay_ms = std::clamp(
          config_.apm.delay.base_ms + playback_delay_ms_.load() +
              capture_delay_ms,
          0, 500);
      apm_->set_stream_delay_ms(total_delay_ms);
    }
    if (config_.apm.agc1.enabled &&
        config_.apm.agc1.mode == "adaptive_analog") {
      // File input and this ALSA example have no mixer-level integration.
      apm_->set_stream_analog_level(127);
    }
    const int result = apm_->ProcessStream(
        apm_frame, stream_config, stream_config, apm_frame);
    if (result != webrtc::AudioProcessing::kNoError) {
      SetError("APM ProcessStream failed: " + std::to_string(result));
      running_.store(false);
      return false;
    }
    for (size_t frame = 0; frame < frames_per_apm_frame_; ++frame) {
      (*processed_left)[frame] = apm_frame[frame * capture_channels];
    }

    if (config_.runtime.input == "file" &&
        config_.runtime.file_input_mode == "batch") {
      size_t written = 0;
      while (written < processed_left->size() && running_.load()) {
        written += recorded_queue_.Write(
            processed_left->data() + written,
            processed_left->size() - written,
            std::chrono::milliseconds(200));
      }
      return written == processed_left->size();
    }
    dropped_recorded_samples_.fetch_add(recorded_queue_.WriteDroppingOldest(
        processed_left->data(), processed_left->size()));
    return true;
  }

  bool ProcessFileFrame(std::vector<int16_t>* apm_frame,
                        std::vector<int16_t>* processed_left,
                        std::chrono::steady_clock::time_point* next_deadline) {
    if (!ProcessCaptureFrame(apm_frame->data(), capture_channels_, 0,
                             processed_left)) {
      return false;
    }
    if (config_.runtime.file_input_mode == "realtime") {
      *next_deadline += std::chrono::milliseconds(kApmFrameMs);
      std::this_thread::sleep_until(*next_deadline);
    }
    return true;
  }

  bool ReadFileWithoutResampling(
      std::chrono::steady_clock::time_point* next_deadline) {
    const size_t sample_count =
        frames_per_apm_frame_ * static_cast<size_t>(capture_channels_);
    const size_t frame_bytes = sample_count * sizeof(int16_t);
    std::vector<unsigned char> bytes(frame_bytes);
    std::vector<int16_t> apm_frame(sample_count);
    std::vector<int16_t> processed_left(frames_per_apm_frame_);

    while (running_.load() && file_bytes_remaining_ > 0) {
      const size_t bytes_to_read = static_cast<size_t>(
          std::min<uint64_t>(frame_bytes, file_bytes_remaining_));
      file_input_.read(reinterpret_cast<char*>(bytes.data()),
                       static_cast<std::streamsize>(bytes_to_read));
      if (file_input_.gcount() != static_cast<std::streamsize>(bytes_to_read)) {
        SetError("input WAV data chunk is truncated");
        running_.store(false);
        return false;
      }
      file_bytes_remaining_ -= bytes_to_read;
      std::fill(apm_frame.begin(), apm_frame.end(), 0);
      for (size_t i = 0; i + 1 < bytes_to_read; i += 2) {
        apm_frame[i / 2] = static_cast<int16_t>(ReadLe16(bytes.data() + i));
      }
      if (!ProcessFileFrame(&apm_frame, &processed_left, next_deadline)) {
        return false;
      }
    }
    return running_.load();
  }

  bool ReadFileWithResampling(
      std::chrono::steady_clock::time_point* next_deadline) {
    const uint64_t source_rate = file_sample_rate_hz_;
    const uint64_t target_rate = config_.audio.capture_sample_rate_hz;
    const uint64_t divisor = std::gcd(source_rate, target_rate);
    const size_t source_base = static_cast<size_t>(source_rate / divisor);
    const size_t target_base = static_cast<size_t>(target_rate / divisor);
    const size_t desired_source_frames = static_cast<size_t>(
        std::max<uint64_t>(64, (source_rate + 99) / 100));
    const size_t block_factor =
        (desired_source_frames + source_base - 1) / source_base;
    const size_t source_block_frames = source_base * block_factor;
    const size_t target_block_frames = target_base * block_factor;
    const size_t channels = static_cast<size_t>(capture_channels_);
    const size_t source_block_bytes =
        source_block_frames * channels * sizeof(int16_t);

    std::vector<std::unique_ptr<webrtc::PushSincResampler>> resamplers;
    std::vector<std::vector<int16_t>> source(
        channels, std::vector<int16_t>(source_block_frames));
    std::vector<std::vector<int16_t>> resampled(
        channels, std::vector<int16_t>(target_block_frames));
    for (size_t channel = 0; channel < channels; ++channel) {
      resamplers.push_back(std::make_unique<webrtc::PushSincResampler>(
          source_block_frames, target_block_frames));
    }

    std::vector<unsigned char> bytes(source_block_bytes);
    std::deque<int16_t> pending;
    const size_t apm_sample_count = frames_per_apm_frame_ * channels;
    std::vector<int16_t> apm_frame(apm_sample_count);
    std::vector<int16_t> processed_left(frames_per_apm_frame_);
    uint64_t source_frames_seen = 0;
    uint64_t target_frames_emitted = 0;

    while (running_.load() && file_bytes_remaining_ > 0) {
      const size_t bytes_to_read = static_cast<size_t>(
          std::min<uint64_t>(source_block_bytes, file_bytes_remaining_));
      file_input_.read(reinterpret_cast<char*>(bytes.data()),
                       static_cast<std::streamsize>(bytes_to_read));
      if (file_input_.gcount() != static_cast<std::streamsize>(bytes_to_read)) {
        SetError("input WAV data chunk is truncated");
        running_.store(false);
        return false;
      }
      file_bytes_remaining_ -= bytes_to_read;
      const size_t source_frames =
          bytes_to_read / (channels * sizeof(int16_t));
      for (auto& channel_samples : source) {
        std::fill(channel_samples.begin(), channel_samples.end(), 0);
      }
      for (size_t frame = 0; frame < source_frames; ++frame) {
        for (size_t channel = 0; channel < channels; ++channel) {
          const size_t byte_offset =
              (frame * channels + channel) * sizeof(int16_t);
          source[channel][frame] =
              static_cast<int16_t>(ReadLe16(bytes.data() + byte_offset));
        }
      }
      for (size_t channel = 0; channel < channels; ++channel) {
        const size_t output_frames = resamplers[channel]->Resample(
            source[channel].data(), source_block_frames,
            resampled[channel].data(), target_block_frames);
        if (output_frames != target_block_frames) {
          SetError("WebRTC file input resampler failed");
          running_.store(false);
          return false;
        }
      }

      source_frames_seen += source_frames;
      const uint64_t expected_target_frames =
          (source_frames_seen * target_rate + source_rate / 2) / source_rate;
      const size_t valid_target_frames = static_cast<size_t>(
          expected_target_frames - target_frames_emitted);
      target_frames_emitted = expected_target_frames;
      if (valid_target_frames > target_block_frames) {
        SetError("file input resampler produced an invalid frame count");
        running_.store(false);
        return false;
      }
      for (size_t frame = 0; frame < valid_target_frames; ++frame) {
        for (size_t channel = 0; channel < channels; ++channel) {
          pending.push_back(resampled[channel][frame]);
        }
      }
      while (pending.size() >= apm_sample_count && running_.load()) {
        for (size_t i = 0; i < apm_sample_count; ++i) {
          apm_frame[i] = pending.front();
          pending.pop_front();
        }
        if (!ProcessFileFrame(&apm_frame, &processed_left, next_deadline)) {
          return false;
        }
      }
    }

    if (!pending.empty() && running_.load()) {
      std::fill(apm_frame.begin(), apm_frame.end(), 0);
      size_t i = 0;
      while (!pending.empty()) {
        apm_frame[i++] = pending.front();
        pending.pop_front();
      }
      if (!ProcessFileFrame(&apm_frame, &processed_left, next_deadline)) {
        return false;
      }
    }
    return running_.load();
  }

  void FileCaptureLoop() {
    auto next_deadline = std::chrono::steady_clock::now();
    const bool success =
        file_sample_rate_hz_ ==
                static_cast<uint32_t>(config_.audio.capture_sample_rate_hz)
            ? ReadFileWithoutResampling(&next_deadline)
            : ReadFileWithResampling(&next_deadline);
    if (success) {
      input_finished_.store(true);
    }
    recorded_queue_.Close();
  }

  bool ConfigurePcm(snd_pcm_t* pcm,
                    bool playback,
                    int sample_rate_hz,
                    int channels,
                    snd_pcm_uframes_t* actual_period_frames,
                    int* actual_channels) {
    snd_pcm_hw_params_t* hw = nullptr;
    snd_pcm_hw_params_alloca(&hw);
    int error = snd_pcm_hw_params_any(pcm, hw);
    if (error >= 0) {
      error = snd_pcm_hw_params_set_access(pcm, hw,
                                           SND_PCM_ACCESS_RW_INTERLEAVED);
    }
    if (error >= 0) {
      error = snd_pcm_hw_params_set_format(pcm, hw, SND_PCM_FORMAT_S16_LE);
    }
    if (error >= 0) {
      // A capture card often exposes only one hardware channel even though
      // the shared configuration defaults to stereo.  Ask ALSA for the
      // nearest supported channel count and use the value selected by ALSA
      // throughout the rest of the pipeline.  Playback remains exact because
      // its queue and monitor contract are explicitly configured by the user.
      if (!playback && actual_channels) {
        unsigned int requested = static_cast<unsigned int>(channels);
        error = snd_pcm_hw_params_set_channels_near(pcm, hw, &requested);
      } else {
        error = snd_pcm_hw_params_set_channels(pcm, hw, channels);
      }
    }
    unsigned int rate = static_cast<unsigned int>(sample_rate_hz);
    if (error >= 0) {
      error = snd_pcm_hw_params_set_rate_near(pcm, hw, &rate, nullptr);
    }
    if (error >= 0 && rate != static_cast<unsigned int>(sample_rate_hz)) {
      SetError("ALSA device cannot use the requested sample rate");
      return false;
    }
    snd_pcm_uframes_t period =
        static_cast<snd_pcm_uframes_t>(sample_rate_hz) *
        config_.buffers.period_ms / 1000;
    if (error >= 0) {
      error = snd_pcm_hw_params_set_period_size_near(pcm, hw, &period, nullptr);
    }
    snd_pcm_uframes_t buffer =
        static_cast<snd_pcm_uframes_t>(sample_rate_hz) *
        config_.buffers.alsa_buffer_ms / 1000;
    if (error >= 0) {
      error = snd_pcm_hw_params_set_buffer_size_near(pcm, hw, &buffer);
    }
    if (error >= 0) {
      error = snd_pcm_hw_params(pcm, hw);
    }
    if (error < 0) {
      SetError(AlsaError(playback ? "configure playback device"
                                  : "configure capture device",
                         error));
      return false;
    }
    error = snd_pcm_hw_params_get_period_size(hw, &period, nullptr);
    if (error < 0) {
      SetError(AlsaError(playback ? "query playback period"
                                  : "query capture period",
                         error));
      return false;
    }
    if (actual_channels) {
      unsigned int selected_channels = 0;
      error = snd_pcm_hw_params_get_channels(hw, &selected_channels);
      if (error < 0 || selected_channels == 0 || selected_channels > 8) {
        SetError(AlsaError(playback ? "query playback channels"
                                    : "query capture channels",
                           error < 0 ? error : -EINVAL));
        return false;
      }
      *actual_channels = static_cast<int>(selected_channels);
    }

    snd_pcm_sw_params_t* sw = nullptr;
    snd_pcm_sw_params_alloca(&sw);
    error = snd_pcm_sw_params_current(pcm, sw);
    if (error >= 0) {
      error = snd_pcm_sw_params_set_avail_min(pcm, sw, period);
    }
    if (error >= 0 && playback) {
      // Start after one period; keep latency low while allowing normal blocking
      // writes to be driven by the ALSA buffer state.
      error = snd_pcm_sw_params_set_start_threshold(pcm, sw, period);
    }
    if (error >= 0) {
      error = snd_pcm_sw_params(pcm, sw);
    }
    if (error >= 0) {
      error = snd_pcm_prepare(pcm);
    }
    if (error < 0) {
      SetError(AlsaError(playback ? "prepare playback device"
                                  : "prepare capture device",
                         error));
      return false;
    }
    *actual_period_frames = period;
    return true;
  }

  bool ReadCapturePeriod(int16_t* samples, snd_pcm_uframes_t frames) {
    snd_pcm_uframes_t offset = 0;
    while (offset < frames && running_.load()) {
      const snd_pcm_sframes_t result =
          snd_pcm_readi(capture_pcm_,
                        samples + offset * capture_channels_,
                        frames - offset);
      if (result > 0) {
        offset += static_cast<snd_pcm_uframes_t>(result);
        continue;
      }
      const int error = snd_pcm_recover(capture_pcm_, static_cast<int>(result), 1);
      if (error < 0 && running_.load()) {
        SetError(AlsaError("capture read", error));
        running_.store(false);
        return false;
      }
    }
    return offset == frames;
  }

  bool WritePlaybackPeriod(const int16_t* samples, snd_pcm_uframes_t frames) {
    snd_pcm_uframes_t offset = 0;
    while (offset < frames && running_.load()) {
      const snd_pcm_sframes_t result =
          snd_pcm_writei(playback_pcm_,
                         samples + offset * config_.audio.playback_channels,
                         frames - offset);
      if (result > 0) {
        offset += static_cast<snd_pcm_uframes_t>(result);
        continue;
      }
      const int error =
          snd_pcm_recover(playback_pcm_, static_cast<int>(result), 1);
      if (error < 0 && running_.load()) {
        SetError(AlsaError("playback write", error));
        running_.store(false);
        return false;
      }
    }
    return offset == frames;
  }

  void PlaybackLoop() {
    const webrtc::StreamConfig stream_config(
        config_.audio.playback_sample_rate_hz,
        config_.audio.playback_channels);
    std::vector<int16_t> apm_frame(playback_samples_per_apm_frame_);
    std::deque<int16_t> processed;
    const size_t period_samples =
        static_cast<size_t>(playback_period_frames_) *
        config_.audio.playback_channels;
    std::vector<int16_t> device_period(period_samples);

    while (running_.load()) {
      while (processed.size() < period_samples) {
        const size_t n = playback_queue_.ReadAvailable(
            apm_frame.data(), apm_frame.size());
        if (n < apm_frame.size()) {
          std::fill(apm_frame.begin() + n, apm_frame.end(), 0);
          playback_underrun_samples_.fetch_add(apm_frame.size() - n);
        }
        const int result = apm_->ProcessReverseStream(
            apm_frame.data(), stream_config, stream_config, apm_frame.data());
        if (result != webrtc::AudioProcessing::kNoError) {
          SetError("APM ProcessReverseStream failed: " +
                   std::to_string(result));
          running_.store(false);
          break;
        }
        processed.insert(processed.end(), apm_frame.begin(), apm_frame.end());
      }
      if (!running_.load()) {
        break;
      }
      for (size_t i = 0; i < period_samples; ++i) {
        device_period[i] = processed.front();
        processed.pop_front();
      }
      if (!WritePlaybackPeriod(device_period.data(), playback_period_frames_)) {
        break;
      }

      snd_pcm_sframes_t delay_frames = 0;
      if (snd_pcm_delay(playback_pcm_, &delay_frames) >= 0) {
        const int delay_ms = static_cast<int>(
            std::max<snd_pcm_sframes_t>(0, delay_frames) * 1000 /
            config_.audio.playback_sample_rate_hz);
        playback_delay_ms_.store(delay_ms);
      }
    }
  }

  void CaptureLoop() {
    const size_t device_period_samples =
        static_cast<size_t>(capture_period_frames_) *
        capture_channels_;
    std::vector<int16_t> device_period(device_period_samples);
    std::vector<int16_t> apm_frame(frames_per_apm_frame_ *
                                   capture_channels_);
    std::vector<int16_t> processed_left(frames_per_apm_frame_);
    std::deque<int16_t> pending_interleaved;

    while (running_.load()) {
      if (!ReadCapturePeriod(device_period.data(), capture_period_frames_)) {
        break;
      }
      pending_interleaved.insert(pending_interleaved.end(),
                                 device_period.begin(), device_period.end());
      while (pending_interleaved.size() >= apm_frame.size() &&
             running_.load()) {
        for (size_t i = 0; i < apm_frame.size(); ++i) {
          apm_frame[i] = pending_interleaved.front();
          pending_interleaved.pop_front();
        }

        snd_pcm_sframes_t capture_delay_frames = 0;
        int capture_delay_ms = 0;
        if (snd_pcm_delay(capture_pcm_, &capture_delay_frames) >= 0) {
          capture_delay_ms = static_cast<int>(
              std::max<snd_pcm_sframes_t>(0, capture_delay_frames) * 1000 /
              config_.audio.capture_sample_rate_hz);
        }
        if (!ProcessCaptureFrame(apm_frame.data(), capture_channels_,
                                 capture_delay_ms, &processed_left)) {
          break;
        }
      }
    }
  }

  void SetError(std::string error) {
    std::lock_guard<std::mutex> lock(error_mutex_);
    last_error_ = std::move(error);
  }

  void CloseDevices() {
    if (capture_pcm_) {
      snd_pcm_close(capture_pcm_);
      capture_pcm_ = nullptr;
    }
    if (playback_pcm_) {
      snd_pcm_close(playback_pcm_);
      playback_pcm_ = nullptr;
    }
  }

  AppConfig config_;
  const size_t frames_per_apm_frame_;
  const size_t playback_samples_per_apm_frame_;
  SampleQueue playback_queue_;
  SampleQueue recorded_queue_;
  snd_pcm_t* playback_pcm_ = nullptr;
  snd_pcm_t* capture_pcm_ = nullptr;
  snd_pcm_uframes_t playback_period_frames_ = 0;
  snd_pcm_uframes_t capture_period_frames_ = 0;
  // Set to the channel count actually selected by ALSA.  This may differ
  // from the requested value when a capture-only device supports mono or
  // stereo exclusively.
  int capture_channels_ = 1;
  std::ifstream file_input_;
  uint32_t file_sample_rate_hz_ = 0;
  uint64_t file_bytes_remaining_ = 0;
  rtc::scoped_refptr<webrtc::AudioProcessing> apm_;
  std::thread playback_thread_;
  std::thread capture_thread_;
  std::atomic<bool> running_{false};
  std::atomic<bool> input_finished_{false};
  std::atomic<int> playback_delay_ms_{0};
  std::atomic<uint64_t> playback_underrun_samples_{0};
  std::atomic<uint64_t> dropped_recorded_samples_{0};
  mutable std::mutex error_mutex_;
  std::string last_error_;
};

AlsaApmBackend::AlsaApmBackend() : AlsaApmBackend(AppConfig()) {}

AlsaApmBackend::AlsaApmBackend(AppConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

AlsaApmBackend::~AlsaApmBackend() = default;

bool AlsaApmBackend::Start() {
  return impl_->Start();
}

void AlsaApmBackend::Stop() {
  impl_->Stop();
}

size_t AlsaApmBackend::WritePlayback(
    const int16_t* samples,
    size_t sample_count,
    std::chrono::milliseconds timeout) {
  return impl_->WritePlayback(samples, sample_count, timeout);
}

size_t AlsaApmBackend::ReadRecorded(
    int16_t* samples,
    size_t sample_count,
    std::chrono::milliseconds timeout) {
  return impl_->ReadRecorded(samples, sample_count, timeout);
}

std::string AlsaApmBackend::last_error() const {
  return impl_->last_error();
}

bool AlsaApmBackend::input_finished() const {
  return impl_->input_finished();
}

uint64_t AlsaApmBackend::playback_underrun_samples() const {
  return impl_->playback_underrun_samples();
}

uint64_t AlsaApmBackend::dropped_recorded_samples() const {
  return impl_->dropped_recorded_samples();
}

}  // namespace apm_example
