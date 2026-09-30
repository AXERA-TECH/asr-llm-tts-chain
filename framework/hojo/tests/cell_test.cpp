// Host-side integration test: real pipeline scheduling/IPC, fake model workers.
#define main audio_pipeline_main
#include "../../audio_pipeline.cpp"
#undef main
#include <cassert>

static void TestChunks() {
  HojoTextChunks chunks;
  assert(chunks.Feed("<thi").empty());
  assert(chunks.Feed("nk>do not speak，</th").empty());
  assert(chunks.Feed("ink>你好\xef\xbc").empty());
  assert(chunks.Feed("\x8c") == std::vector<std::string>{"你好，"});
  assert(chunks.Feed("3.").empty());
  assert(chunks.Feed("14 is pi. tail") == std::vector<std::string>{"3.14 is pi."});
  assert(chunks.Feed("", true) == std::vector<std::string>{"tail"});
  assert(chunks.Feed("，。...\n", true).empty());
  HojoTextChunks bounded;
  assert(bounded.Feed(std::string(205, 'a')).size() == 2);
  assert(bounded.Feed("", true) == std::vector<std::string>{"aaaaa"});
  // Arbitrary byte boundaries must preserve all UTF-8 text and flush the tail.
  const std::string text = "中文，English!剩余无标点";
  HojoTextChunks bytes;
  std::vector<std::string> actual;
  for (char c : text) {
    for (auto& chunk : bytes.Feed(std::string(1, c))) actual.push_back(chunk);
  }
  for (auto& chunk : bytes.Feed("", true)) actual.push_back(chunk);
  assert((actual == std::vector<std::string>{"中文，", "English!", "剩余无标点"}));
}

int main(int argc, char** argv) {
  assert(argc == 4);
  std::signal(SIGPIPE, SIG_IGN);
  TestChunks();
  const fs::path root(argv[1]);
  const std::string scenario(argv[3]);
  apm_example::AppConfig c;
  c.asr.enabled = true;
  c.asr.library = (root / "fake-asr.so").string();
  c.llm.enabled = true;
  c.llm.runner = argv[2];
  c.llm.stream_tokens = false;  // TTS must still receive incremental frames.
  c.llm.output_file = (root / "llm.txt").string();
  c.llm.npu_lock_file = (root / "npu.lock").string();
  c.llm.queue_capacity = 1;
  c.hojo_tts.enabled = scenario != "disabled";
  c.hojo_tts.runner = argv[2];
  c.hojo_tts.output_dir = (root / "tts with spaces").string();
  c.hojo_tts.queue_capacity = 1; // A tiny preparation queue must not deadlock.
  c.output.temp_dir = (root / "segments").string();
  c.output.mp3_dir = (root / "mp3").string();
  c.output.text_file = (root / "asr.txt").string();
  Npu2CellArbiter arbiter(c.llm.npu_lock_file);
  std::size_t plays = 0;
  ResultWorker worker(c, &arbiter, [&](std::vector<int16_t> pcm) {
    assert(pcm.size() == 480); // Mock generates 240 samples at 24 kHz.
    ++plays;
    std::ofstream(root / "events", std::ios::app) << "PLAY\n";
  });
  Segment s;
  s.samples.resize(1600, 100);
  s.wall_start = WallClock::now();
  s.wall_end = s.wall_start + std::chrono::milliseconds(100);
  if (scenario == "parts") {
    s.parts = {{0, 800, "speaker0"}, {800, 1600, "speaker1"}};
    worker.Submit(s);
  } else {
    worker.Submit(s);
    s.wall_start += std::chrono::seconds(1);
    s.wall_end += std::chrono::seconds(1);
    worker.Submit(s);
  }
  worker.Close();
  const std::size_t expected = (scenario == "disabled" || scenario == "disconnect") ? 0 :
      (scenario == "error" || scenario == "empty" ||
       scenario == "prepare_error" || scenario == "synth_error" ? 2 : 4);
  assert(plays == expected);
  std::cout << "PASS " << scenario << " playback chunks=" << plays << '\n';
}
