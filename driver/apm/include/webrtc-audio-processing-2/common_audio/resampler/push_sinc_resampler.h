/* Copyright (c) 2013 The WebRTC project authors. BSD-style license. */
#ifndef COMMON_AUDIO_RESAMPLER_PUSH_SINC_RESAMPLER_H_
#define COMMON_AUDIO_RESAMPLER_PUSH_SINC_RESAMPLER_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>

#include "common_audio/resampler/sinc_resampler.h"

namespace webrtc {

class PushSincResampler : public SincResamplerCallback {
 public:
  PushSincResampler(size_t source_frames, size_t destination_frames);
  ~PushSincResampler() override;

  PushSincResampler(const PushSincResampler&) = delete;
  PushSincResampler& operator=(const PushSincResampler&) = delete;

  size_t Resample(const int16_t* source,
                  size_t source_frames,
                  int16_t* destination,
                  size_t destination_capacity);
  size_t Resample(const float* source,
                  size_t source_frames,
                  float* destination,
                  size_t destination_capacity);

  static float AlgorithmicDelaySeconds(int source_rate_hz) {
    return 1.f / source_rate_hz * SincResampler::kKernelSize / 2;
  }

 protected:
  void Run(size_t frames, float* destination) override;

 private:
  friend class PushSincResamplerTest;
  SincResampler* get_resampler_for_testing() { return resampler_.get(); }

  std::unique_ptr<SincResampler> resampler_;
  std::unique_ptr<float[]> float_buffer_;
  const float* source_ptr_;
  const int16_t* source_ptr_int_;
  const size_t destination_frames_;
  bool first_pass_;
  size_t source_available_;
};

}  // namespace webrtc

#endif  // COMMON_AUDIO_RESAMPLER_PUSH_SINC_RESAMPLER_H_
