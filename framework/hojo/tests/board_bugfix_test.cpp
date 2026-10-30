// AX650 integration probe: actual Qwen/Hojo models and cached notices.
#define main audio_pipeline_main
#include "../../audio_pipeline.cpp"
#undef main
#include <cassert>

int main(int argc, char** argv) {
  assert(argc == 2);
  std::signal(SIGPIPE, SIG_IGN);
  apm_example::AppConfig c;
  std::string error;
  assert(apm_example::LoadAppConfig(argv[1], &c, &error));
  assert(c.llm.min_asr_characters == 5 && c.llm.max_asr_punctuation_ratio == 0.5f);
  assert(c.hojo_tts.min_chinese_characters == 20 && c.hojo_tts.min_english_words == 10);
  assert(c.llm.system_prompt.find("简洁干练") != std::string::npos);
  const auto root = fs::path("/tmp/asr-bugfix-board-validation");
  fs::create_directories(root);
  c.hojo_tts.output_dir = (root / "tts").string();
  c.llm.output_file = (root / "llm.txt").string();
  Npu2CellArbiter arbiter(c.llm.npu_lock_file);
  {
    // A real PREPARE can race cancellation; no stale SYNTH/PCM may follow.
    c.hojo_tts.enabled = true;
    c.hojo_tts.min_chinese_characters = c.hojo_tts.min_english_words = 1;
    c.hojo_tts.backlog_threshold = 2;
    PlaybackMixer mixer;
    std::vector<int16_t> delivered;
    int plays = 0;
    HojoTtsWorker tts(c, [&](std::vector<int16_t> pcm) {
      delivered = pcm;
      mixer.AppendTts(std::move(pcm));
      ++plays;
    }, &mixer);
    assert(tts.active());
    for (const std::string language : {"en", "zh"}) {
      mixer.ClearTts();
      mixer.AppendTts(std::vector<int16_t>(48000, 123));
      const bool chinese = language == "zh";
      const auto voice = chinese ? c.hojo_tts.chinese_voice : c.hojo_tts.english_voice;
      std::vector<int16_t> expected;
      assert(LoadWavForPlayback(fs::path(c.hojo_tts.notice_dir) /
          (language + "-" + std::to_string(voice) + ".wav"),
          c.hojo_tts.playback_gain, &expected, &error));
      auto lease = arbiter.Acquire();
      tts.Begin("[" + language + "]");
      const std::string text = chinese ? "一，二，三，四。" : "One, two, three, four.";
      tts.Token(text);
      tts.Finish(text, true);
      assert(delivered == expected);
      assert(mixer.PendingFrames() == 1);  // Old one-second PCM was discarded.
      std::vector<int16_t> output(expected.size());
      mixer.Mix(&output, false);
      assert(output == expected && mixer.PendingFrames() == 0);
      std::cout << "PASS cached " << language << " notice voice=" << voice << '\n';
    }
    assert(plays == 2);
    // A subsequent normal reply must still synthesize successfully.
    auto lease = arbiter.Acquire();
    tts.Begin("[zh]");
    tts.Token("你好。");
    tts.Finish("你好。", true);
    assert(plays == 3 && !delivered.empty());
    std::cout << "PASS real Hojo synthesis after overflow\n";
  }
  c.hojo_tts.enabled = false;
  c.llm.enabled = true;
  for (const bool translation : {true, false}) {
    c.llm.translation_mode = translation;
    int requests = 0, replies = 0;
    LlmWorker llm(c, [&](const std::string&) { ++requests; }, {},
        [&](const std::string& text, bool success) {
          assert(success && !text.empty());
          assert(text.find("<think>") == std::string::npos);
          assert(text.find("</think>") == std::string::npos);
          ++replies;
          std::cout << "REPLY " << (translation ? "translation" : "assistant") << ": " << text << '\n';
        });
    assert(llm.active());
    auto lease = arbiter.Acquire();
    llm.ProcessAsrResult("[ASR_ERROR] sensevoice unavailable\n[time] speaker0@0-1 ，。！？…——“”\n[time] speaker0@0-1 好\n[time] speaker0@0-1 一二三四五\n[time] speaker0@0-1 你好啊，！？");
    assert(requests == 0 && replies == 0);
    llm.ProcessAsrResult(translation ? "Hello, how are you?" : "请问你是谁？");
    if (translation) llm.ProcessAsrResult("今天天气很好。");
    assert(requests == (translation ? 2 : 1) && requests == replies);
    std::cout << "PASS real Qwen " << (translation ? "translation" : "assistant") << '\n';
  }
}
