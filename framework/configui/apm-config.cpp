#include "apm-config.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace apm_example {
namespace {

std::string Trim(const std::string& value) {
  const auto first = std::find_if_not(value.begin(), value.end(), [](char c) {
    return std::isspace(static_cast<unsigned char>(c));
  });
  const auto last = std::find_if_not(value.rbegin(), value.rend(), [](char c) {
                      return std::isspace(static_cast<unsigned char>(c));
                    }).base();
  return first < last ? std::string(first, last) : std::string();
}

std::string StripComment(const std::string& line) {
  bool single_quote = false;
  bool double_quote = false;
  for (size_t i = 0; i < line.size(); ++i) {
    if (line[i] == '\'' && !double_quote) {
      single_quote = !single_quote;
    } else if (line[i] == '"' && !single_quote &&
               (i == 0 || line[i - 1] != '\\')) {
      double_quote = !double_quote;
    } else if (line[i] == '#' && !single_quote && !double_quote) {
      return line.substr(0, i);
    }
  }
  return line;
}

std::string Unquote(std::string value) {
  value = Trim(value);
  if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
    const std::string quoted = value.substr(1, value.size() - 2);
    value.clear();
    for (std::size_t i = 0; i < quoted.size(); ++i) {
      if (quoted[i] != '\\' || i + 1 >= quoted.size()) {
        value.push_back(quoted[i]);
        continue;
      }
      const char escaped = quoted[++i];
      if (escaped == 'n') value.push_back('\n');
      else if (escaped == 'r') value.push_back('\r');
      else if (escaped == 't') value.push_back('\t');
      else if (escaped == '"') value.push_back('"');
      else if (escaped == '\\') value.push_back('\\');
      else { value.push_back('\\'); value.push_back(escaped); }
    }
  } else if (value.size() >= 2 && value.front() == '\'' &&
             value.back() == '\'') {
    value = value.substr(1, value.size() - 2);
  }
  return value;
}

bool ParseYaml(const std::string& path,
               std::map<std::string, std::string>* values,
               std::string* error) {
  std::ifstream input(path);
  if (!input) {
    *error = "cannot open config file: " + path;
    return false;
  }

  struct Parent {
    int indent;
    std::string key;
  };
  std::vector<Parent> parents;
  std::string raw;
  int line_number = 0;
  while (std::getline(input, raw)) {
    ++line_number;
    const std::string line = StripComment(raw);
    if (Trim(line).empty()) {
      continue;
    }
    const size_t first = line.find_first_not_of(' ');
    if (first == std::string::npos || line.find('\t', 0) != std::string::npos) {
      *error = "tabs are not allowed in YAML indentation at line " +
               std::to_string(line_number);
      return false;
    }
    const int indent = static_cast<int>(first);
    const std::string content = line.substr(first);
    const size_t colon = content.find(':');
    if (colon == std::string::npos) {
      *error = "expected key: value at line " + std::to_string(line_number);
      return false;
    }
    const std::string key = Trim(content.substr(0, colon));
    const std::string value = Trim(content.substr(colon + 1));
    if (key.empty()) {
      *error = "empty YAML key at line " + std::to_string(line_number);
      return false;
    }
    while (!parents.empty() && indent <= parents.back().indent) {
      parents.pop_back();
    }
    std::string full_key;
    for (const auto& parent : parents) {
      if (!full_key.empty()) full_key += '.';
      full_key += parent.key;
    }
    if (!full_key.empty()) full_key += '.';
    full_key += key;
    if (value.empty()) {
      parents.push_back({indent, key});
    } else {
      if (!values->emplace(full_key, Unquote(value)).second) {
        *error = "duplicate YAML key: " + full_key;
        return false;
      }
    }
  }
  return true;
}

bool GetString(const std::map<std::string, std::string>& values,
               const std::string& key,
               std::string* output,
               std::string* error) {
  const auto it = values.find(key);
  if (it == values.end()) return true;
  if (it->second.empty()) {
    *error = key + " must not be empty";
    return false;
  }
  *output = it->second;
  return true;
}

bool GetStringAllowEmpty(const std::map<std::string, std::string>& values,
                         const std::string& key,
                         std::string* output,
                         std::string* /*error*/) {
  const auto it = values.find(key);
  if (it == values.end()) return true;
  *output = it->second;
  return true;
}

bool GetBool(const std::map<std::string, std::string>& values,
             const std::string& key,
             bool* output,
             std::string* error) {
  const auto it = values.find(key);
  if (it == values.end()) return true;
  std::string value = it->second;
  std::transform(value.begin(), value.end(), value.begin(), [](char c) {
    return std::tolower(static_cast<unsigned char>(c));
  });
  if (value == "true") {
    *output = true;
  } else if (value == "false") {
    *output = false;
  } else {
    *error = key + " must be true or false";
    return false;
  }
  return true;
}

template <typename T>
bool GetNumber(const std::map<std::string, std::string>& values,
               const std::string& key,
               T* output,
               std::string* error) {
  const auto it = values.find(key);
  if (it == values.end()) return true;
  char* end = nullptr;
  errno = 0;
  const double number = std::strtod(it->second.c_str(), &end);
  if (errno != 0 || end == it->second.c_str() || *end != '\0') {
    *error = key + " must be numeric";
    return false;
  }
  *output = static_cast<T>(number);
  return true;
}

bool GetNumber(const std::map<std::string, std::string>& values,
               const std::string& key,
               int* output,
               std::string* error) {
  const auto it = values.find(key);
  if (it == values.end()) return true;
  char* end = nullptr;
  errno = 0;
  const double number = std::strtod(it->second.c_str(), &end);
  if (errno != 0 || end == it->second.c_str() || *end != '\0' ||
      !std::isfinite(number) || std::floor(number) != number ||
      number < static_cast<double>(std::numeric_limits<int>::min()) ||
      number > static_cast<double>(std::numeric_limits<int>::max())) {
    *error = key + " must be an integer-valued number";
    return false;
  }
  *output = static_cast<int>(number);
  return true;
}

bool IsNativeRate(int rate) {
  return rate == 8000 || rate == 16000 || rate == 32000 || rate == 48000;
}

}  // namespace

bool ValidateApmConfig(const AppConfig& c, std::string* error);

bool ValidateAppConfig(const AppConfig& c, std::string* error) {
  if (!ValidateApmConfig(c, error)) return false;
  if (c.fastenhance.npu_core != 0 || c.vad.npu_core != 0 ||
      c.asr.npu_core != 1 || c.hojo_tts.npu_core != 1) {
    *error = "FastEnhance/TEN-VAD must use NPU1; SenseVoice/Hojo must use the NPU2 slot";
    return false;
  }
  if (c.vad.hop_ms <= 0 || c.vad.hop_ms > 100 || c.vad.threshold < 0.0f ||
      c.vad.threshold > 1.0f || c.vad.min_speech_ms <= 0 ||
      c.vad.min_segment_seconds <= 0 || c.vad.max_segment_seconds <= 0 ||
      c.vad.trailing_silence_seconds <= 0 ||
      c.vad.input_buffer_ms <= 0) {
    *error = "VAD hop/threshold/segment values are invalid";
    return false;
  }
  if (c.vad.max_segment_seconds <=
      c.vad.min_segment_seconds + c.vad.trailing_silence_seconds) {
    *error = "VAD max accumulation must be greater than min accumulation plus trailing silence";
    return false;
  }
  if (c.campplus.min_cluster_size < 0 ||
      c.campplus.merge_cosine < -1.0f || c.campplus.merge_cosine > 1.0f ||
      c.campplus.speaker_match_threshold < -1.0f ||
      c.campplus.speaker_match_threshold > 1.0f ||
      c.campplus.min_num_speakers < 1 ||
      c.campplus.max_num_speakers < c.campplus.min_num_speakers ||
      c.campplus.model_path.empty() || c.campplus.speaker_table_path.empty()) {
    *error = "CampPlus clustering/model values are invalid";
    return false;
  }
  if (c.output.max_mp3_storage_mb < 0 || c.asr.worker_threads <= 0) {
    *error = "output storage and ASR worker values are invalid";
    return false;
  }
  const std::set<std::string> thinking_modes = {
      "default", "enabled", "disabled"};
  if (c.llm.model_path.empty() || c.llm.runner.empty() ||
      c.llm.zh_to_en_prompt.empty() || c.llm.en_to_zh_prompt.empty() ||
      c.llm.output_file.empty() || c.llm.npu_lock_file.empty() ||
      c.llm.dynamic_load_pool_size <= 0 || c.llm.memory_guard_floor_mb < 0 ||
      c.llm.max_tokens <= 0 || c.llm.max_tokens > 2047 ||
      !std::isfinite(c.llm.temperature) || c.llm.temperature < 0.0f ||
      !std::isfinite(c.llm.top_p) || c.llm.top_p <= 0.0f ||
      c.llm.top_p > 1.0f || c.llm.top_k < 0 ||
      !std::isfinite(c.llm.repetition_penalty) ||
      c.llm.repetition_penalty <= 0.0f ||
      !std::isfinite(c.llm.frequency_penalty) ||
      c.llm.frequency_penalty < -2.0f || c.llm.frequency_penalty > 2.0f ||
      !std::isfinite(c.llm.presence_penalty) ||
      c.llm.presence_penalty < -2.0f || c.llm.presence_penalty > 2.0f ||
      c.llm.queue_capacity < 0 ||
      !thinking_modes.count(c.llm.thinking_mode)) {
    *error = "Qwen3 model/loading/generation values are invalid";
    return false;
  }
  if (c.hojo_tts.runner.empty() || c.hojo_tts.model_path.empty() ||
      c.hojo_tts.output_dir.empty() ||
      c.hojo_tts.english_voice < 0 || c.hojo_tts.english_voice > 12 ||
      c.hojo_tts.chinese_voice < 13 || c.hojo_tts.chinese_voice > 14 ||
      c.hojo_tts.max_new_tokens <= 0 ||
      c.hojo_tts.max_new_tokens > 2175 || c.hojo_tts.queue_capacity < 0 ||
      !std::isfinite(c.hojo_tts.playback_gain) ||
      c.hojo_tts.playback_gain < 0.0f || c.hojo_tts.playback_gain > 4.0f) {
    *error = "Hojo TTS paths/voices/generation/playback values are invalid";
    return false;
  }
  if (c.hojo_tts.enabled && !c.llm.enabled) {
    *error = "Hojo TTS requires the Qwen3 LLM FIFO to be enabled";
    return false;
  }
  return true;
}

bool ValidateApmConfig(const AppConfig& c, std::string* error) {
  if (!error) return false;
  if (c.runtime.input != "alsa" && c.runtime.input != "file") {
    *error = "runtime.input must be alsa or file";
    return false;
  }
  if (c.runtime.file_input_mode != "realtime" &&
      c.runtime.file_input_mode != "batch") {
    *error = "runtime.file_input_mode must be realtime or batch";
    return false;
  }
  if (c.runtime.input == "file" && c.runtime.input_wav.empty()) {
    *error = "runtime.input_wav must not be empty for file input";
    return false;
  }
  if (!IsNativeRate(c.audio.playback_sample_rate_hz) ||
      !IsNativeRate(c.audio.capture_sample_rate_hz)) {
    *error = "S16 APM sample rates must be one of 8000, 16000, 32000, 48000";
    return false;
  }
  if (c.audio.playback_sample_rate_hz != c.audio.capture_sample_rate_hz) {
    *error = "S16 APM requires playback and capture sample rates to match";
    return false;
  }
  if (c.audio.playback_channels < 1 || c.audio.playback_channels > 8 ||
      c.audio.capture_channels < 1 || c.audio.capture_channels > 8) {
    *error = "playback/capture channels must be in [1, 8]";
    return false;
  }
  if (c.buffers.period_ms <= 0 ||
      c.buffers.alsa_buffer_ms < c.buffers.period_ms ||
      c.buffers.application_buffer_ms <= 0) {
    *error = "buffer values are invalid";
    return false;
  }
  if (c.apm.delay.base_ms < 0 || c.apm.delay.base_ms > 500) {
    *error = "apm.delay.base_ms must be in [0, 500]";
    return false;
  }
  if (c.apm.aec.mode != "aec3" && c.apm.aec.mode != "mobile") {
    *error = "apm.aec.mode must be aec3 or mobile";
    return false;
  }
  const std::set<std::string> ns_levels = {"low", "moderate", "high", "very_high"};
  if (!ns_levels.count(c.apm.ns.level)) {
    *error = "apm.ns.level must be low, moderate, high, or very_high";
    return false;
  }
  const std::set<std::string> agc1_modes = {
      "adaptive_analog", "adaptive_digital", "fixed_digital"};
  if (!agc1_modes.count(c.apm.agc1.mode)) {
    *error = "apm.agc1.mode is invalid";
    return false;
  }
  if (c.apm.agc1.target_level_dbfs < 0 ||
      c.apm.agc1.target_level_dbfs > 31 ||
      c.apm.agc1.compression_gain_db < 0 ||
      c.apm.agc1.compression_gain_db > 90) {
    *error = "AGC1 target must be [0,31] and compression gain [0,90]";
    return false;
  }
  if (c.apm.agc2.fixed_gain_db < 0 || c.apm.agc2.fixed_gain_db >= 50 ||
      c.apm.agc2.adaptive_headroom_db < 0 ||
      c.apm.agc2.adaptive_max_gain_db <= 0 ||
      c.apm.agc2.adaptive_initial_gain_db < 0 ||
      c.apm.agc2.adaptive_max_gain_change_db_per_second <= 0 ||
      c.apm.agc2.adaptive_max_output_noise_level_dbfs > 0) {
    *error = "AGC2 values are outside their supported ranges";
    return false;
  }
  if (!std::isfinite(c.apm.gain_adjustment.pre_gain_factor) ||
      !std::isfinite(c.apm.gain_adjustment.post_gain_factor) ||
      c.apm.gain_adjustment.pre_gain_factor <= 0 ||
      c.apm.gain_adjustment.post_gain_factor <= 0 ||
      c.apm.gain_adjustment.pre_gain_factor > 1000 ||
      c.apm.gain_adjustment.post_gain_factor > 1000) {
    *error = "capture pre/post gain factors must be finite and in (0, 1000]";
    return false;
  }
  return true;
}

bool LoadAppConfig(const std::string& path,
                   AppConfig* c,
                   std::string* error) {
  if (!c || !error) return false;
  std::map<std::string, std::string> v;
  if (!ParseYaml(path, &v, error)) return false;

  const std::set<std::string> known = {
      "audio.playback.device", "audio.playback.sample_rate_hz",
      "audio.playback.channels", "audio.capture.device",
      "audio.capture.sample_rate_hz", "audio.capture.channels",
      "buffers.period_ms", "buffers.alsa_buffer_ms",
      "buffers.application_buffer_ms", "apm.aec.enabled", "apm.aec.mode",
      "apm.aec.multi_channel_render", "apm.aec.multi_channel_capture",
      "apm.delay.base_ms", "apm.ns.enabled", "apm.ns.level",
      "apm.agc1.enabled", "apm.agc1.mode", "apm.agc1.target_level_dbfs",
      "apm.agc1.compression_gain_db", "apm.agc1.limiter",
      "apm.agc2.enabled", "apm.agc2.fixed_gain_db",
      "apm.agc2.adaptive_digital", "apm.agc2.adaptive_headroom_db",
      "apm.agc2.adaptive_max_gain_db", "apm.agc2.adaptive_initial_gain_db",
      "apm.agc2.adaptive_max_gain_change_db_per_second",
      "apm.agc2.adaptive_max_output_noise_level_dbfs",
      "apm.gain_adjustment.enabled",
      "apm.gain_adjustment.pre_gain_factor",
      "apm.gain_adjustment.post_gain_factor",
      "apm.high_pass_filter_enabled",
      "fastenhance.enabled", "fastenhance.model_path",
      "fastenhance.library", "fastenhance.npu_core",
      "fastenhance.monitor_enabled",
      "vad.enabled", "vad.model_path", "vad.library", "vad.npu_core",
      "vad.input_buffer_ms", "vad.hop_ms", "vad.threshold", "vad.min_speech_ms",
      "vad.min_segment_seconds", "vad.max_segment_seconds",
      "vad.trailing_silence_seconds",
      "campplus.enabled", "campplus.logging_enabled", "campplus.model_path",
      "campplus.speaker_table_path", "campplus.min_cluster_size",
      "campplus.merge_cosine", "campplus.min_num_speakers",
      "campplus.max_num_speakers", "campplus.speaker_match_threshold",
      "asr.enabled", "asr.model_path", "asr.model_type", "asr.language",
      "asr.library", "asr.npu_core", "asr.worker_threads",
      "llm.enabled", "llm.model_path", "llm.runner", "llm.system_prompt",
      "llm.translation_mode", "llm.zh_to_en_prompt", "llm.en_to_zh_prompt",
      "llm.dynamic_load", "llm.dynamic_load_pool_size",
      "llm.memory_guard", "llm.memory_guard_floor_mb", "llm.max_tokens",
      "llm.temperature", "llm.top_p", "llm.top_k",
      "llm.repetition_penalty", "llm.frequency_penalty",
      "llm.presence_penalty", "llm.thinking_mode", "llm.reset_context",
      "llm.stream_tokens", "llm.queue_capacity", "llm.output_file",
      "llm.npu_lock_file",
      "hojo_tts.enabled", "hojo_tts.runner", "hojo_tts.model_path",
      "hojo_tts.english_voice", "hojo_tts.chinese_voice",
      "hojo_tts.max_new_tokens",
      "hojo_tts.queue_capacity", "hojo_tts.keep_wav",
      "hojo_tts.output_dir", "hojo_tts.playback_gain", "hojo_tts.npu_core",
      "output.temp_dir", "output.text_file", "output.mp3_dir",
      "output.max_mp3_storage_mb", "output.mp3_encoder",
      "runtime.input", "runtime.input_wav", "runtime.file_input_mode",
      "runtime.apm_binary",
      "runtime.log_level", "runtime.playback_enabled"};
  for (const auto& item : v) {
    if (!known.count(item.first)) {
      *error = "unknown config key: " + item.first;
      return false;
    }
  }

#define GET_STRING(key, field) if (!GetString(v, key, &field, error)) return false
#define GET_OPTIONAL_STRING(key, field) \
  if (!GetStringAllowEmpty(v, key, &field, error)) return false
#define GET_BOOL(key, field) if (!GetBool(v, key, &field, error)) return false
#define GET_NUM(key, field) if (!GetNumber(v, key, &field, error)) return false
  GET_STRING("audio.playback.device", c->audio.playback_device);
  GET_NUM("audio.playback.sample_rate_hz", c->audio.playback_sample_rate_hz);
  GET_NUM("audio.playback.channels", c->audio.playback_channels);
  GET_STRING("audio.capture.device", c->audio.capture_device);
  GET_NUM("audio.capture.sample_rate_hz", c->audio.capture_sample_rate_hz);
  GET_NUM("audio.capture.channels", c->audio.capture_channels);
  GET_NUM("buffers.period_ms", c->buffers.period_ms);
  GET_NUM("buffers.alsa_buffer_ms", c->buffers.alsa_buffer_ms);
  GET_NUM("buffers.application_buffer_ms", c->buffers.application_buffer_ms);
  GET_BOOL("apm.aec.enabled", c->apm.aec.enabled);
  GET_STRING("apm.aec.mode", c->apm.aec.mode);
  GET_BOOL("apm.aec.multi_channel_render", c->apm.aec.multi_channel_render);
  GET_BOOL("apm.aec.multi_channel_capture", c->apm.aec.multi_channel_capture);
  GET_NUM("apm.delay.base_ms", c->apm.delay.base_ms);
  GET_BOOL("apm.ns.enabled", c->apm.ns.enabled);
  GET_STRING("apm.ns.level", c->apm.ns.level);
  GET_BOOL("apm.agc1.enabled", c->apm.agc1.enabled);
  GET_STRING("apm.agc1.mode", c->apm.agc1.mode);
  GET_NUM("apm.agc1.target_level_dbfs", c->apm.agc1.target_level_dbfs);
  GET_NUM("apm.agc1.compression_gain_db", c->apm.agc1.compression_gain_db);
  GET_BOOL("apm.agc1.limiter", c->apm.agc1.limiter);
  GET_BOOL("apm.agc2.enabled", c->apm.agc2.enabled);
  GET_NUM("apm.agc2.fixed_gain_db", c->apm.agc2.fixed_gain_db);
  GET_BOOL("apm.agc2.adaptive_digital", c->apm.agc2.adaptive_digital);
  GET_NUM("apm.agc2.adaptive_headroom_db", c->apm.agc2.adaptive_headroom_db);
  GET_NUM("apm.agc2.adaptive_max_gain_db", c->apm.agc2.adaptive_max_gain_db);
  GET_NUM("apm.agc2.adaptive_initial_gain_db",
          c->apm.agc2.adaptive_initial_gain_db);
  GET_NUM("apm.agc2.adaptive_max_gain_change_db_per_second",
          c->apm.agc2.adaptive_max_gain_change_db_per_second);
  GET_NUM("apm.agc2.adaptive_max_output_noise_level_dbfs",
          c->apm.agc2.adaptive_max_output_noise_level_dbfs);
  GET_BOOL("apm.gain_adjustment.enabled",
           c->apm.gain_adjustment.enabled);
  GET_NUM("apm.gain_adjustment.pre_gain_factor",
          c->apm.gain_adjustment.pre_gain_factor);
  GET_NUM("apm.gain_adjustment.post_gain_factor",
          c->apm.gain_adjustment.post_gain_factor);
  GET_BOOL("apm.high_pass_filter_enabled",
           c->apm.high_pass_filter_enabled);
  GET_BOOL("fastenhance.enabled", c->fastenhance.enabled);
  GET_STRING("fastenhance.model_path", c->fastenhance.model_path);
  GET_OPTIONAL_STRING("fastenhance.library", c->fastenhance.library);
  GET_NUM("fastenhance.npu_core", c->fastenhance.npu_core);
  GET_BOOL("fastenhance.monitor_enabled", c->fastenhance.monitor_enabled);
  GET_BOOL("vad.enabled", c->vad.enabled);
  GET_STRING("vad.model_path", c->vad.model_path);
  GET_OPTIONAL_STRING("vad.library", c->vad.library);
  GET_NUM("vad.npu_core", c->vad.npu_core);
  GET_NUM("vad.input_buffer_ms", c->vad.input_buffer_ms);
  GET_NUM("vad.hop_ms", c->vad.hop_ms);
  GET_NUM("vad.threshold", c->vad.threshold);
  GET_NUM("vad.min_speech_ms", c->vad.min_speech_ms);
  GET_NUM("vad.min_segment_seconds", c->vad.min_segment_seconds);
  GET_NUM("vad.max_segment_seconds", c->vad.max_segment_seconds);
  GET_NUM("vad.trailing_silence_seconds", c->vad.trailing_silence_seconds);
  GET_BOOL("campplus.enabled", c->campplus.enabled);
  GET_BOOL("campplus.logging_enabled", c->campplus.logging_enabled);
  GET_STRING("campplus.model_path", c->campplus.model_path);
  GET_STRING("campplus.speaker_table_path", c->campplus.speaker_table_path);
  GET_NUM("campplus.min_cluster_size", c->campplus.min_cluster_size);
  GET_NUM("campplus.merge_cosine", c->campplus.merge_cosine);
  GET_NUM("campplus.min_num_speakers", c->campplus.min_num_speakers);
  GET_NUM("campplus.max_num_speakers", c->campplus.max_num_speakers);
  GET_NUM("campplus.speaker_match_threshold", c->campplus.speaker_match_threshold);
  GET_BOOL("asr.enabled", c->asr.enabled);
  GET_STRING("asr.model_path", c->asr.model_path);
  GET_STRING("asr.model_type", c->asr.model_type);
  GET_STRING("asr.language", c->asr.language);
  GET_OPTIONAL_STRING("asr.library", c->asr.library);
  GET_NUM("asr.npu_core", c->asr.npu_core);
  GET_NUM("asr.worker_threads", c->asr.worker_threads);
  GET_BOOL("llm.enabled", c->llm.enabled);
  GET_STRING("llm.model_path", c->llm.model_path);
  GET_STRING("llm.runner", c->llm.runner);
  GET_OPTIONAL_STRING("llm.system_prompt", c->llm.system_prompt);
  GET_BOOL("llm.translation_mode", c->llm.translation_mode);
  GET_STRING("llm.zh_to_en_prompt", c->llm.zh_to_en_prompt);
  GET_STRING("llm.en_to_zh_prompt", c->llm.en_to_zh_prompt);
  GET_BOOL("llm.dynamic_load", c->llm.dynamic_load);
  GET_NUM("llm.dynamic_load_pool_size", c->llm.dynamic_load_pool_size);
  GET_BOOL("llm.memory_guard", c->llm.memory_guard);
  GET_NUM("llm.memory_guard_floor_mb", c->llm.memory_guard_floor_mb);
  GET_NUM("llm.max_tokens", c->llm.max_tokens);
  GET_NUM("llm.temperature", c->llm.temperature);
  GET_NUM("llm.top_p", c->llm.top_p);
  GET_NUM("llm.top_k", c->llm.top_k);
  GET_NUM("llm.repetition_penalty", c->llm.repetition_penalty);
  GET_NUM("llm.frequency_penalty", c->llm.frequency_penalty);
  GET_NUM("llm.presence_penalty", c->llm.presence_penalty);
  GET_STRING("llm.thinking_mode", c->llm.thinking_mode);
  GET_BOOL("llm.reset_context", c->llm.reset_context);
  GET_BOOL("llm.stream_tokens", c->llm.stream_tokens);
  GET_NUM("llm.queue_capacity", c->llm.queue_capacity);
  GET_STRING("llm.output_file", c->llm.output_file);
  GET_STRING("llm.npu_lock_file", c->llm.npu_lock_file);
  GET_BOOL("hojo_tts.enabled", c->hojo_tts.enabled);
  GET_STRING("hojo_tts.runner", c->hojo_tts.runner);
  GET_STRING("hojo_tts.model_path", c->hojo_tts.model_path);
  GET_NUM("hojo_tts.english_voice", c->hojo_tts.english_voice);
  GET_NUM("hojo_tts.chinese_voice", c->hojo_tts.chinese_voice);
  GET_NUM("hojo_tts.max_new_tokens", c->hojo_tts.max_new_tokens);
  GET_NUM("hojo_tts.queue_capacity", c->hojo_tts.queue_capacity);
  GET_BOOL("hojo_tts.keep_wav", c->hojo_tts.keep_wav);
  GET_STRING("hojo_tts.output_dir", c->hojo_tts.output_dir);
  GET_NUM("hojo_tts.playback_gain", c->hojo_tts.playback_gain);
  GET_NUM("hojo_tts.npu_core", c->hojo_tts.npu_core);
  GET_STRING("output.temp_dir", c->output.temp_dir);
  GET_STRING("output.text_file", c->output.text_file);
  GET_STRING("output.mp3_dir", c->output.mp3_dir);
  GET_NUM("output.max_mp3_storage_mb", c->output.max_mp3_storage_mb);
  GET_STRING("output.mp3_encoder", c->output.mp3_encoder);
  GET_STRING("runtime.input", c->runtime.input);
  GET_OPTIONAL_STRING("runtime.input_wav", c->runtime.input_wav);
  GET_STRING("runtime.file_input_mode", c->runtime.file_input_mode);
  GET_STRING("runtime.apm_binary", c->runtime.apm_binary);
  GET_STRING("runtime.log_level", c->runtime.log_level);
  GET_BOOL("runtime.playback_enabled", c->runtime.playback_enabled);
#undef GET_STRING
#undef GET_OPTIONAL_STRING
#undef GET_BOOL
#undef GET_NUM
  return ValidateAppConfig(*c, error);
}

bool LoadApmConfig(const std::string& path,
                   AppConfig* c,
                   std::string* error) {
  if (!c || !error) return false;
  std::map<std::string, std::string> v;
  if (!ParseYaml(path, &v, error)) return false;

  const std::set<std::string> owned = {
      "audio.playback.device", "audio.playback.sample_rate_hz",
      "audio.playback.channels", "audio.capture.device",
      "audio.capture.sample_rate_hz", "audio.capture.channels",
      "buffers.period_ms", "buffers.alsa_buffer_ms",
      "buffers.application_buffer_ms", "apm.aec.enabled", "apm.aec.mode",
      "apm.aec.multi_channel_render", "apm.aec.multi_channel_capture",
      "apm.delay.base_ms", "apm.ns.enabled", "apm.ns.level",
      "apm.agc1.enabled", "apm.agc1.mode", "apm.agc1.target_level_dbfs",
      "apm.agc1.compression_gain_db", "apm.agc1.limiter",
      "apm.agc2.enabled", "apm.agc2.fixed_gain_db",
      "apm.agc2.adaptive_digital", "apm.agc2.adaptive_headroom_db",
      "apm.agc2.adaptive_max_gain_db", "apm.agc2.adaptive_initial_gain_db",
      "apm.agc2.adaptive_max_gain_change_db_per_second",
      "apm.agc2.adaptive_max_output_noise_level_dbfs",
      "apm.gain_adjustment.enabled",
      "apm.gain_adjustment.pre_gain_factor",
      "apm.gain_adjustment.post_gain_factor",
      "apm.high_pass_filter_enabled", "runtime.input", "runtime.input_wav",
      "runtime.file_input_mode", "runtime.playback_enabled"};
  for (const auto& item : v) {
    const bool apm_owned = item.first.rfind("audio.", 0) == 0 ||
                           item.first.rfind("buffers.", 0) == 0 ||
                           item.first.rfind("apm.", 0) == 0;
    if (apm_owned && !owned.count(item.first)) {
      *error = "unknown APM config key: " + item.first;
      return false;
    }
  }

#define APM_GET_STRING(key, field) if (!GetString(v, key, &field, error)) return false
#define APM_GET_OPTIONAL_STRING(key, field) \
  if (!GetStringAllowEmpty(v, key, &field, error)) return false
#define APM_GET_BOOL(key, field) if (!GetBool(v, key, &field, error)) return false
#define APM_GET_NUM(key, field) if (!GetNumber(v, key, &field, error)) return false
  APM_GET_STRING("audio.playback.device", c->audio.playback_device);
  APM_GET_NUM("audio.playback.sample_rate_hz", c->audio.playback_sample_rate_hz);
  APM_GET_NUM("audio.playback.channels", c->audio.playback_channels);
  APM_GET_STRING("audio.capture.device", c->audio.capture_device);
  APM_GET_NUM("audio.capture.sample_rate_hz", c->audio.capture_sample_rate_hz);
  APM_GET_NUM("audio.capture.channels", c->audio.capture_channels);
  APM_GET_NUM("buffers.period_ms", c->buffers.period_ms);
  APM_GET_NUM("buffers.alsa_buffer_ms", c->buffers.alsa_buffer_ms);
  APM_GET_NUM("buffers.application_buffer_ms", c->buffers.application_buffer_ms);
  APM_GET_BOOL("apm.aec.enabled", c->apm.aec.enabled);
  APM_GET_STRING("apm.aec.mode", c->apm.aec.mode);
  APM_GET_BOOL("apm.aec.multi_channel_render", c->apm.aec.multi_channel_render);
  APM_GET_BOOL("apm.aec.multi_channel_capture", c->apm.aec.multi_channel_capture);
  APM_GET_NUM("apm.delay.base_ms", c->apm.delay.base_ms);
  APM_GET_BOOL("apm.ns.enabled", c->apm.ns.enabled);
  APM_GET_STRING("apm.ns.level", c->apm.ns.level);
  APM_GET_BOOL("apm.agc1.enabled", c->apm.agc1.enabled);
  APM_GET_STRING("apm.agc1.mode", c->apm.agc1.mode);
  APM_GET_NUM("apm.agc1.target_level_dbfs", c->apm.agc1.target_level_dbfs);
  APM_GET_NUM("apm.agc1.compression_gain_db", c->apm.agc1.compression_gain_db);
  APM_GET_BOOL("apm.agc1.limiter", c->apm.agc1.limiter);
  APM_GET_BOOL("apm.agc2.enabled", c->apm.agc2.enabled);
  APM_GET_NUM("apm.agc2.fixed_gain_db", c->apm.agc2.fixed_gain_db);
  APM_GET_BOOL("apm.agc2.adaptive_digital", c->apm.agc2.adaptive_digital);
  APM_GET_NUM("apm.agc2.adaptive_headroom_db", c->apm.agc2.adaptive_headroom_db);
  APM_GET_NUM("apm.agc2.adaptive_max_gain_db", c->apm.agc2.adaptive_max_gain_db);
  APM_GET_NUM("apm.agc2.adaptive_initial_gain_db", c->apm.agc2.adaptive_initial_gain_db);
  APM_GET_NUM("apm.agc2.adaptive_max_gain_change_db_per_second", c->apm.agc2.adaptive_max_gain_change_db_per_second);
  APM_GET_NUM("apm.agc2.adaptive_max_output_noise_level_dbfs", c->apm.agc2.adaptive_max_output_noise_level_dbfs);
  APM_GET_BOOL("apm.gain_adjustment.enabled", c->apm.gain_adjustment.enabled);
  APM_GET_NUM("apm.gain_adjustment.pre_gain_factor", c->apm.gain_adjustment.pre_gain_factor);
  APM_GET_NUM("apm.gain_adjustment.post_gain_factor", c->apm.gain_adjustment.post_gain_factor);
  APM_GET_BOOL("apm.high_pass_filter_enabled", c->apm.high_pass_filter_enabled);
  APM_GET_STRING("runtime.input", c->runtime.input);
  APM_GET_OPTIONAL_STRING("runtime.input_wav", c->runtime.input_wav);
  APM_GET_STRING("runtime.file_input_mode", c->runtime.file_input_mode);
  APM_GET_BOOL("runtime.playback_enabled", c->runtime.playback_enabled);
#undef APM_GET_STRING
#undef APM_GET_OPTIONAL_STRING
#undef APM_GET_BOOL
#undef APM_GET_NUM
  return ValidateApmConfig(*c, error);
}

}  // namespace apm_example
