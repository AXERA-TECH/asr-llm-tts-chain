#pragma once

#include <cctype>
#include <string>
#include <vector>

// Incremental UTF-8 sentence splitter. A chunk is immutable once emitted, so
// its tokenizer/embeddings can be prepared while Qwen continues decoding.
class HojoTextChunks {
 public:
  std::vector<std::string> Feed(const std::string& delta, bool final = false) {
    input_ += delta;
    std::vector<std::string> result;
    while (!input_.empty()) {
      bool tag = false;
      for (const std::string marker : {"<think>", "</think>"}) {
        if (input_.rfind(marker, 0) == 0) {
          thinking_ = marker == "<think>";
          input_.erase(0, marker.size());
          tag = true;
          break;
        }
        if (!final && marker.compare(0, input_.size(), input_) == 0)
          return result;
      }
      if (tag) continue;
      const unsigned char c = input_[0];
      const std::size_t width = c < 0x80 ? 1 : (c < 0xe0 ? 2 : (c < 0xf0 ? 3 : 4));
      if (input_.size() < width) {
        if (final) input_.clear();
        break;
      }
      // Delay a trailing period until the next byte distinguishes a decimal.
      if (!thinking_ && c == '.' && input_.size() == 1 && !final) break;
      const std::string character = input_.substr(0, width);
      const bool decimal = c == '.' && !chunk_.empty() &&
          std::isdigit(static_cast<unsigned char>(chunk_.back())) &&
          input_.size() > 1 && std::isdigit(static_cast<unsigned char>(input_[1]));
      input_.erase(0, width);
      if (thinking_) continue;
      chunk_ += character;
      ++characters_;
      const bool punctuation = (!decimal && character == ".") ||
          character == "," || character == "!" || character == "?" ||
          character == ";" || character == ":" || character == "\n" ||
          character == "。" || character == "，" || character == "！" ||
          character == "？" || character == "；" || character == "：" ||
          character == "、";
      // Prefer whitespace for long English sentences; bound unpunctuated CJK.
      if (punctuation || characters_ >= 100 ||
          (characters_ >= 80 && character == " "))
        Emit(&result);
    }
    if (final) Emit(&result);
    return result;
  }

 private:
  void Emit(std::vector<std::string>* result) {
    const auto begin = chunk_.find_first_not_of(" \t\r\n");
    if (begin != std::string::npos) {
      auto text = chunk_.substr(begin, chunk_.find_last_not_of(" \t\r\n") - begin + 1);
      // Suppress runs of punctuation, including split ellipses.
      bool content = false;
      for (std::size_t i = 0; i < text.size();) {
        const unsigned char c = text[i];
        const std::size_t width = c < 0x80 ? 1 : (c < 0xe0 ? 2 : (c < 0xf0 ? 3 : 4));
        const auto ch = text.substr(i, width);
        if (width == 1 ? std::isalnum(c) != 0 :
            ch != "。" && ch != "，" && ch != "！" && ch != "？" &&
            ch != "；" && ch != "：" && ch != "、" && ch != "…") content = true;
        i += width;
      }
      if (content) result->push_back(std::move(text));
    }
    chunk_.clear();
    characters_ = 0;
  }
  std::string input_, chunk_;
  std::size_t characters_ = 0;
  bool thinking_ = false;
};
