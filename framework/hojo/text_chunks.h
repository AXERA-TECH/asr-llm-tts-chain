#pragma once

#include <cctype>
#include <string>
#include <vector>
#include "../text_utils.h"

// Incremental UTF-8 sentence splitter. A chunk is immutable once emitted, so
// its tokenizer/embeddings can be prepared while Qwen continues decoding.
class HojoTextChunks {
 public:
  explicit HojoTextChunks(int chinese_minimum = 20, int english_minimum = 10,
                          std::string language = "")
      : chinese_minimum_(chinese_minimum), english_minimum_(english_minimum),
        language_(std::move(language)) {}

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
      std::size_t offset = 0;
      const auto cp = text_utils::Next(character, &offset);
      const bool content = !text_utils::IsSpace(cp) && !text_utils::IsPunctuation(cp);
      if (content) ++characters_;
      chinese_ = chinese_ || text_utils::IsChinese(cp);
      if (content && !text_utils::IsChinese(cp)) {
        if (!in_word_) ++words_;
        in_word_ = true;
      } else if (character != "'" && character != "’" && character != "-" && !decimal) {
        in_word_ = false;
      }
      const bool punctuation = (!decimal && character == ".") ||
          character == "," || character == "!" || character == "?" ||
          character == ";" || character == ":" ||
          character == "。" || character == "，" || character == "！" ||
          character == "？" || character == "；" || character == "：" ||
          character == "、" || character == "…";
      const bool use_chinese = language_ == "[zh]" ||
          (language_.empty() && chinese_);
      // Reach the minimum first, then wait for punctuation. Never force a
      // split in the middle of a sentence, even for long unpunctuated text.
      if (punctuation && (use_chinese ? characters_ >= chinese_minimum_ :
                                      words_ >= english_minimum_))
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
      const auto stats = text_utils::AnalyzeAsr(text);
      if (stats.characters > stats.punctuation) result->push_back(std::move(text));
    }
    chunk_.clear();
    characters_ = words_ = 0;
    in_word_ = chinese_ = false;
  }
  std::string input_, chunk_;
  int characters_ = 0, words_ = 0;
  int chinese_minimum_, english_minimum_;
  std::string language_;
  bool in_word_ = false, chinese_ = false;
  bool thinking_ = false;
};
