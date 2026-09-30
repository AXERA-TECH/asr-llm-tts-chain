#pragma once

#include "configui/apm-config.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace campplus {

struct LabeledPart {
  std::size_t begin = 0;
  std::size_t end = 0;
  std::string speaker_id;
};

// A streaming CAMPPlus session. Push() is fed only VAD-voiced samples and
// MarkJoinPoint() identifies removed-silence boundaries in that sample stream.
// Finalize() clusters all 1.5 s/0.75 s embeddings, snaps speaker boundaries
// to the nearest join point, and updates SpeakerTable.
class Session {
 public:
  explicit Session(const apm_example::AppConfig::Campplus& config,
                   std::function<void()> lock = {},
                   std::function<void()> unlock = {});
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  bool available() const;
  void MarkJoinPoint(std::size_t sample_offset);
  void Push(const std::vector<int16_t>& samples);
  std::vector<LabeledPart> Finalize(std::size_t sample_count,
                                    const std::string& segment_name = {},
                                    const std::vector<std::size_t>& join_points = {});
  void Reset();

 private:
  struct Impl;
  Impl* impl_ = nullptr;
};

}  // namespace campplus

#ifdef CAMPPLUS_CLUSTER_TEST_API
namespace campplus {
std::vector<int> ClusterForTest(
    const std::vector<std::vector<float>>& embeddings,
    const apm_example::AppConfig::Campplus& config,
    const std::string& segment_name);
std::vector<int16_t> MaskJoinsForTest(
    const std::vector<int16_t>& samples, std::size_t window_begin,
    const std::vector<std::size_t>& join_points);
std::size_t NearestJoinForTest(
    std::size_t boundary, const std::vector<std::size_t>& join_points);
}  // namespace campplus
#endif
