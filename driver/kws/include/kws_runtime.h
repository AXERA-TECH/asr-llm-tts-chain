#pragma once

#include "configui/apm-config.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace kws {

struct Detection {
  std::string label;
  float confidence = 0.0f;
  std::uint64_t frame = 0;
};

class Runtime {
 public:
  Runtime(const apm_example::AppConfig::Kws& config,
          std::function<void()> lock, std::function<void()> unlock);
  ~Runtime();
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  bool enabled() const;
  std::vector<Detection> Process48k(const std::vector<std::int16_t>& samples);
  void Reset();
  void Report() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kws
