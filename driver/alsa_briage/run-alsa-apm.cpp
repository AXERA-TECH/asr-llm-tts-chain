#include <chrono>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <condition_variable>
#include <iostream>
#include <deque>
#include <memory>
#include <mutex>
#include <poll.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "alsa-apm-backend.h"

namespace {

constexpr int kApmFrameMs = 10;
volatile std::sig_atomic_t g_stop_requested = 0;

class MonitorInput {
 public:
  explicit MonitorInput(size_t buffer_frames)
      : capacity_(buffer_frames) {
    thread_ = std::thread(&MonitorInput::ReadLoop, this);
  }

  ~MonitorInput() {
    Stop();
  }

  // Consume an exact FIFO batch on the dedicated monitor playback thread.
  size_t ReadExact(int16_t* output, size_t frame_count,
                   std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    std::unique_lock<std::mutex> lock(mutex_);
    if (!not_empty_.wait_until(lock, deadline, [&] {
          return closed_ || samples_.size() >= frame_count;
        }) || closed_ || samples_.size() < frame_count) {
      return 0;
    }
    for (size_t i = 0; i < frame_count; ++i) {
      output[i] = samples_.front();
      samples_.pop_front();
    }
    not_full_.notify_one();
    return frame_count;
  }

  uint64_t frames_read() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return frames_read_;
  }

  bool failed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return failed_;
  }

  bool closed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return closed_;
  }

  void Stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    not_empty_.notify_all();
    not_full_.notify_all();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

 private:
  void ReadLoop() {
    std::vector<int16_t> input(512);
    while (!g_stop_requested) {
      pollfd descriptor{STDIN_FILENO, POLLIN, 0};
      const int ready = poll(&descriptor, 1, 100);
      if (ready == 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) return;
        continue;
      }
      if (ready < 0) {
        if (errno == EINTR) continue;
        std::lock_guard<std::mutex> lock(mutex_);
        failed_ = true;
        not_empty_.notify_all();
        return;
      }
      const ssize_t read_bytes =
          read(STDIN_FILENO, input.data(), input.size() * sizeof(int16_t));
      const ssize_t read_frames =
          read_bytes / static_cast<ssize_t>(sizeof(int16_t));
      if (read_frames <= 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        failed_ = true;
        not_empty_.notify_all();
        return;
      }
      const size_t count = static_cast<size_t>(read_frames);
      std::unique_lock<std::mutex> lock(mutex_);
      if (first_read_) {
        std::cerr << "Monitor stdin received first " << count << " frames\n";
        first_read_ = false;
      }
      frames_read_ += count;
      not_full_.wait(lock, [&] {
        return closed_ || samples_.size() <= capacity_ - count;
      });
      if (closed_) {
        return;
      }
      samples_.insert(samples_.end(), input.begin(), input.begin() + count);
      not_empty_.notify_all();
    }
  }

  const size_t capacity_;
  std::deque<int16_t> samples_;
  mutable std::mutex mutex_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  std::thread thread_;
  uint64_t frames_read_ = 0;
  bool first_read_ = true;
  bool closed_ = false;
  bool failed_ = false;
};

void HandleSignal(int) {
  g_stop_requested = 1;
}

void PrintUsage(const char* program) {
  std::cerr << "Usage: " << program
            << " [--stdout] [--monitor-stdin] [sdk-config.yaml]\n\n"
            << "Monitors the APM-processed capture signal through the configured "
               "ALSA playback device.\n"
            << "--stdout writes processed mono S16_LE capture to stdout for the "
               "post-APM SDK pipeline.\n";
}

}  // namespace

int main(int argc, char** argv) {
  bool stdout_mode = false;
  bool monitor_stdin_requested = false;
  std::string config_path;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--stdout") {
      stdout_mode = true;
    } else if (argument == "--monitor-stdin") {
      monitor_stdin_requested = true;
    } else if (config_path.empty()) {
      config_path = argument;
    } else {
      PrintUsage(argv[0]);
      return EXIT_FAILURE;
    }
  }
  if (config_path.empty()) {
    config_path = "config/sdk-config.yaml";
  }
  apm_example::AppConfig config;
  std::string config_error;
  if (!apm_example::LoadApmConfig(config_path, &config, &config_error)) {
    std::cerr << "Configuration error: " << config_error << '\n';
    return EXIT_FAILURE;
  }
  if (stdout_mode) {
    config.runtime.playback_enabled = false;
  }
  if (!config.runtime.playback_enabled && monitor_stdin_requested) {
    config.runtime.playback_enabled = true;
  }

  apm_example::AlsaApmBackend backend(config);
  if (!backend.Start()) {
    std::cerr << "Backend start failed: " << backend.last_error() << '\n';
    return EXIT_FAILURE;
  }

  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);

  // Keep the process boundary on the APM input's 10 ms frames. FastEnhance's
  // 512-sample hop is assembled by the downstream pipeline.
  const size_t capture_frames_per_chunk =
      static_cast<size_t>(config.audio.capture_sample_rate_hz) * kApmFrameMs /
      1000;
  const size_t playback_channels =
      static_cast<size_t>(config.audio.playback_channels);
  std::vector<int16_t> captured(capture_frames_per_chunk);
  std::vector<int16_t> playback(capture_frames_per_chunk * playback_channels);
  uint64_t monitored_frames = 0;
  bool failed = false;
  const bool monitor_stdin = stdout_mode && monitor_stdin_requested;
  const size_t monitor_buffer_frames =
      std::max<size_t>(1024,
                       static_cast<size_t>(config.audio.playback_sample_rate_hz) *
                           config.buffers.application_buffer_ms / 1000);
  std::unique_ptr<MonitorInput> monitor_input;
  if (monitor_stdin) {
    monitor_input = std::make_unique<MonitorInput>(monitor_buffer_frames);
  }
  std::atomic<uint64_t> monitor_playback_underruns{0};
  std::atomic<bool> monitor_playback_failed{false};
  std::thread monitor_playback_thread;
  if (monitor_stdin) {
    monitor_playback_thread = std::thread([&] {
      const size_t playback_frames =
          static_cast<size_t>(config.audio.playback_sample_rate_hz) *
          kApmFrameMs / 1000;
      const size_t channels = playback_channels;
      std::vector<int16_t> monitor_frame(playback_frames, 0);
      std::vector<int16_t> playback_frame(playback_frames * channels);
      std::vector<int16_t> last_frame(playback_frames, 0);
      bool have_last_frame = false;
      while (!g_stop_requested) {
        const size_t count = monitor_input->ReadExact(
            monitor_frame.data(), playback_frames,
            std::chrono::milliseconds(200));
        if (count != playback_frames) {
          if (monitor_input->closed() || monitor_input->failed()) {
            monitor_playback_failed.store(true);
            return;
          }
          monitor_playback_underruns.fetch_add(1);
          if (have_last_frame) {
            monitor_frame = last_frame;
          }
        } else {
          last_frame = monitor_frame;
          have_last_frame = true;
        }
        for (size_t frame = 0; frame < playback_frames; ++frame) {
          for (size_t channel = 0; channel < channels; ++channel) {
            playback_frame[frame * channels + channel] = monitor_frame[frame];
          }
        }
        size_t written = 0;
        while (written < playback_frame.size() && !g_stop_requested) {
          const size_t n = backend.WritePlayback(
              playback_frame.data() + written,
              playback_frame.size() - written,
              std::chrono::milliseconds(200));
          if (n == 0) {
            monitor_playback_failed.store(true);
            return;
          }
          written += n;
        }
      }
    });
  }
  if (stdout_mode) {
    std::cerr << "APM " << config.runtime.input << " input stdout mode started";
    if (config.runtime.input == "file") {
      std::cerr << ": " << config.runtime.input_wav << " ("
                << config.runtime.file_input_mode << ")";
    }
    std::cerr << '\n';
  } else {
    std::cout << "Monitoring APM-processed capture on "
              << config.audio.playback_device << " ("
              << config.audio.playback_sample_rate_hz << " Hz, "
              << config.audio.playback_channels << " channels). Press Ctrl+C to "
                 "stop.\n";
  }

  while (!g_stop_requested) {
    if (monitor_playback_failed.load()) {
      failed = true;
      break;
    }
    const size_t captured_frames = backend.ReadRecorded(
        captured.data(), captured.size(), std::chrono::milliseconds(200));
    if (captured_frames == 0) {
      const std::string error = backend.last_error();
      if (!error.empty()) {
        std::cerr << "Capture failed: " << error << '\n';
        failed = true;
        break;
      }
      if (backend.input_finished()) {
        break;
      }
      continue;
    }

    // ReadRecorded() exposes the processed left capture channel. Duplicate
    // it to every configured playback channel for interleaved ALSA playback.
    for (size_t frame = 0; frame < captured_frames; ++frame) {
      for (size_t channel = 0; channel < playback_channels; ++channel) {
        playback[frame * playback_channels + channel] = captured[frame];
      }
    }

    if (stdout_mode) {
      std::cout.write(reinterpret_cast<const char*>(captured.data()),
                      static_cast<std::streamsize>(
                          captured_frames * sizeof(int16_t)));
      std::cout.flush();
      if (!std::cout.good()) failed = true;
    }

    // In monitor mode the dedicated monitor playback thread is the only
    // producer. Otherwise enqueue the selected capture signal here.
    if (config.runtime.playback_enabled && !failed && !monitor_stdin) {
      const size_t playback_samples = captured_frames * playback_channels;
      size_t written = 0;
      while (written < playback_samples && !g_stop_requested) {
        const size_t n = backend.WritePlayback(
            playback.data() + written, playback_samples - written,
            std::chrono::milliseconds(200));
        if (n == 0) {
          const std::string error = backend.last_error();
          std::cerr << "Playback failed or timed out";
          if (!error.empty()) {
            std::cerr << ": " << error;
          }
          std::cerr << '\n';
          failed = true;
          break;
        }
        written += n;
      }
    }
    if (failed) {
      break;
    }
    monitored_frames += captured_frames;
  }

  if (monitor_input) {
    monitor_input->Stop();
  }
  if (monitor_playback_thread.joinable()) {
    monitor_playback_thread.join();
  }
  backend.Stop();

  std::ostream& status_stream = stdout_mode ? std::cerr : std::cout;
  status_stream << "Stopped after "
            << static_cast<double>(monitored_frames) /
                   config.audio.capture_sample_rate_hz
            << " s, monitor stdin received "
            << (monitor_input ? monitor_input->frames_read() : 0)
            << " frames, playback zero-filled "
            << backend.playback_underrun_samples() << " samples, dropped "
            << backend.dropped_recorded_samples() << " recorded samples, monitor "
            << monitor_playback_underruns.load() << " playback underruns\n";
  return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
