#ifndef EXAMPLES_ALSA_APM_BACKEND_H_
#define EXAMPLES_ALSA_APM_BACKEND_H_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "apm-config.h"

namespace apm_example {

// A configurable full-duplex ALSA backend for interleaved signed 16-bit PCM.
// ALSA capture is processed by APM with the channel count selected by ALSA
// (mono and stereo are both supported), after which only its first channel is
// exposed by ReadRecorded().
//
// ALSA period sizes are intentionally not exposed by this API. Internally the
// backend repackets the device periods into the 10 ms frames required by APM.
class AlsaApmBackend {
 public:
  AlsaApmBackend();
  explicit AlsaApmBackend(AppConfig config);
  ~AlsaApmBackend();

  AlsaApmBackend(const AlsaApmBackend&) = delete;
  AlsaApmBackend& operator=(const AlsaApmBackend&) = delete;

  bool Start();
  void Stop();

  // Writes interleaved samples to the playback queue. sample_count is a count
  // of int16_t values, not audio frames.
  size_t WritePlayback(
      const int16_t* samples,
      size_t sample_count,
      std::chrono::milliseconds timeout = std::chrono::milliseconds(1000));

  // Reads the first channel extracted after APM capture processing.
  size_t ReadRecorded(
      int16_t* samples,
      size_t sample_count,
      std::chrono::milliseconds timeout = std::chrono::milliseconds(1000));

  std::string last_error() const;
  // True after a finite file input has been completely processed and drained.
  bool input_finished() const;
  uint64_t playback_underrun_samples() const;
  uint64_t dropped_recorded_samples() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace apm_example

#endif  // EXAMPLES_ALSA_APM_BACKEND_H_
