#include "qwen3_runner.hpp"

#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <unistd.h>

namespace {

struct Options {
  qwen3::ModelLoadOptions model;
  qwen3::GenerationOptions generation;
  bool stream_tokens = false;
  int response_fd = 3;
  std::string lock_file = "/tmp/ax-audio-sdk-npu2.lock";
  bool use_lock = true;
};

bool ReadAll(int fd, void* data, std::size_t size) {
  auto* output = static_cast<unsigned char*>(data);
  while (size > 0) {
    const ssize_t count = read(fd, output, size);
    if (count == 0) return false;
    if (count < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    output += count;
    size -= static_cast<std::size_t>(count);
  }
  return true;
}

bool WriteAll(int fd, const void* data, std::size_t size) {
  const auto* input = static_cast<const unsigned char*>(data);
  while (size > 0) {
    const ssize_t count = write(fd, input, size);
    if (count < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    input += count;
    size -= static_cast<std::size_t>(count);
  }
  return true;
}

bool WriteFrame(int fd, char type, const std::string& payload) {
  const uint32_t size = static_cast<uint32_t>(payload.size());
  return WriteAll(fd, &type, sizeof(type)) &&
         WriteAll(fd, &size, sizeof(size)) &&
         WriteAll(fd, payload.data(), payload.size());
}

const char* RequireValue(int argc, char** argv, int* index) {
  if (*index + 1 >= argc)
    throw std::invalid_argument(std::string("missing value for ") + argv[*index]);
  return argv[++*index];
}

int ParseInt(const char* value, const char* name) {
  std::size_t consumed = 0;
  const std::string text(value);
  const int parsed = std::stoi(text, &consumed);
  if (consumed != text.size())
    throw std::invalid_argument(std::string("invalid integer for ") + name);
  return parsed;
}

float ParseFloat(const char* value, const char* name) {
  std::size_t consumed = 0;
  const std::string text(value);
  const float parsed = std::stof(text, &consumed);
  if (consumed != text.size())
    throw std::invalid_argument(std::string("invalid number for ") + name);
  return parsed;
}

Options ParseArgs(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--model") {
      options.model.model_dir = RequireValue(argc, argv, &i);
    } else if (arg == "--system") {
      options.model.system_prompt = RequireValue(argc, argv, &i);
    } else if (arg == "--dynamic-load") {
      options.model.dynamic_load = true;
    } else if (arg == "--dynamic-pool") {
      options.model.dynamic_load_pool_size =
          ParseInt(RequireValue(argc, argv, &i), "--dynamic-pool");
    } else if (arg == "--no-memory-guard") {
      options.model.memory_guard = false;
    } else if (arg == "--memory-floor-mb") {
      options.model.memory_guard_floor_mb =
          ParseInt(RequireValue(argc, argv, &i), "--memory-floor-mb");
    } else if (arg == "--max-tokens") {
      options.generation.max_tokens =
          ParseInt(RequireValue(argc, argv, &i), "--max-tokens");
    } else if (arg == "--temperature") {
      options.generation.temperature =
          ParseFloat(RequireValue(argc, argv, &i), "--temperature");
    } else if (arg == "--top-p") {
      options.generation.top_p =
          ParseFloat(RequireValue(argc, argv, &i), "--top-p");
    } else if (arg == "--top-k") {
      options.generation.top_k =
          ParseInt(RequireValue(argc, argv, &i), "--top-k");
    } else if (arg == "--repetition-penalty") {
      options.generation.repetition_penalty =
          ParseFloat(RequireValue(argc, argv, &i), "--repetition-penalty");
    } else if (arg == "--frequency-penalty") {
      options.generation.frequency_penalty =
          ParseFloat(RequireValue(argc, argv, &i), "--frequency-penalty");
    } else if (arg == "--presence-penalty") {
      options.generation.presence_penalty =
          ParseFloat(RequireValue(argc, argv, &i), "--presence-penalty");
    } else if (arg == "--thinking") {
      const std::string value = RequireValue(argc, argv, &i);
      if (value == "enabled") options.generation.thinking = qwen3::ThinkingMode::kEnabled;
      else if (value == "disabled") options.generation.thinking = qwen3::ThinkingMode::kDisabled;
      else if (value == "default") options.generation.thinking = qwen3::ThinkingMode::kDefault;
      else throw std::invalid_argument("--thinking must be default, enabled, or disabled");
    } else if (arg == "--keep-context") {
      options.generation.reset_context = false;
    } else if (arg == "--stream-tokens") {
      options.stream_tokens = true;
    } else if (arg == "--response-fd") {
      options.response_fd = ParseInt(RequireValue(argc, argv, &i), "--response-fd");
    } else if (arg == "--lock-file") {
      options.lock_file = RequireValue(argc, argv, &i);
    } else if (arg == "--no-lock") {
      options.use_lock = false;
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  if (options.model.model_dir.empty())
    throw std::invalid_argument("--model is required");
  return options;
}

class FileLock {
 public:
  FileLock(const std::string& path, bool enabled) : enabled_(enabled) {
    if (!enabled_) return;
    fd_ = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0666);
    if (fd_ < 0) throw std::runtime_error("cannot open NPU2 lock: " + path);
  }
  ~FileLock() { if (fd_ >= 0) close(fd_); }
  void Lock() {
    if (!enabled_) return;
    while (flock(fd_, LOCK_EX) != 0) {
      if (errno != EINTR) throw std::runtime_error("cannot lock NPU2");
    }
  }
  void Unlock() { if (enabled_) flock(fd_, LOCK_UN); }
 private:
  bool enabled_ = true;
  int fd_ = -1;
};

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGPIPE, SIG_IGN);
  int response_fd = 3;
  try {
    Options options = ParseArgs(argc, argv);
    response_fd = options.response_fd;
    FileLock npu2_lock(options.lock_file, options.use_lock);
    qwen3::Qwen3Runner runner;
    runner.Load(options.model);
    if (!WriteFrame(response_fd, 'R', "ready")) return 1;

    uint32_t prompt_size = 0;
    while (ReadAll(STDIN_FILENO, &prompt_size, sizeof(prompt_size))) {
      if (prompt_size == 0 || prompt_size > 16U * 1024U * 1024U) {
        if (!WriteFrame(response_fd, 'E', "invalid prompt size")) break;
        continue;
      }
      std::string prompt(prompt_size, '\0');
      if (!ReadAll(STDIN_FILENO, prompt.data(), prompt.size())) break;
      try {
        qwen3::GenerationOptions generation = options.generation;
        if (options.stream_tokens) {
          generation.on_token = [response_fd](const std::string& token) {
            WriteFrame(response_fd, 'T', token);
          };
        }
        npu2_lock.Lock();
        qwen3::InferenceResult result;
        try {
          result = runner.Infer(prompt, generation);
        } catch (...) {
          npu2_lock.Unlock();
          throw;
        }
        npu2_lock.Unlock();
        std::cerr << "[llm] prompt_tokens=" << result.prompt_tokens
                  << " completion_tokens=" << result.completion_tokens
                  << " ttft_ms=" << result.ttft_ms
                  << " prefill_token_s=" << result.prefill_tokens_per_second
                  << " decode_token_s=" << result.decode_tokens_per_second << '\n';
        if (!WriteFrame(response_fd, 'D', result.text)) break;
      } catch (const std::exception& error) {
        if (!WriteFrame(response_fd, 'E', error.what())) break;
      }
    }
    return 0;
  } catch (const std::exception& error) {
    WriteFrame(response_fd, 'E', error.what());
    std::cerr << "[llm] worker error: " << error.what() << '\n';
    return 1;
  }
}
