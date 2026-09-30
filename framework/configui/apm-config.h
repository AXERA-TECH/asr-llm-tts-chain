#ifndef EXAMPLES_CONFIGUI_APM_CONFIG_H_
#define EXAMPLES_CONFIGUI_APM_CONFIG_H_

#include <string>

namespace apm_example {

struct AppConfig {
  struct Audio {
    std::string playback_device = "hw:1,3";
    int playback_sample_rate_hz = 48000;
    int playback_channels = 2;
    std::string capture_device = "hw:1,2";
    int capture_sample_rate_hz = 48000;
    int capture_channels = 2;
  } audio;

  struct Buffers {
    int period_ms = 10;
    int alsa_buffer_ms = 40;
    int application_buffer_ms = 1000;
  } buffers;

  struct Apm {
    struct Aec {
      bool enabled = true;
      std::string mode = "aec3";  // "aec3" or "mobile".
      bool multi_channel_render = true;
      bool multi_channel_capture = true;
    } aec;

    struct Delay {
      // Added to the playback and capture ALSA FIFO delays.
      int base_ms = 0;
    } delay;

    struct NoiseSuppression {
      bool enabled = true;
      std::string level = "high";  // low, moderate, high, very_high.
    } ns;

    struct GainController1 {
      bool enabled = false;
      std::string mode = "fixed_digital";
      int target_level_dbfs = 3;
      int compression_gain_db = 9;
      bool limiter = true;
    } agc1;

    struct GainController2 {
      bool enabled = true;
      float fixed_gain_db = 0.0f;
      bool adaptive_digital = false;
      float adaptive_headroom_db = 5.0f;
      float adaptive_max_gain_db = 50.0f;
      float adaptive_initial_gain_db = 15.0f;
      float adaptive_max_gain_change_db_per_second = 6.0f;
      float adaptive_max_output_noise_level_dbfs = -50.0f;
    } agc2;

    struct CaptureLevelAdjustment {
      bool enabled = true;
      // Linear amplitude factors. The UI displays and edits these in dB.
      float pre_gain_factor = 1.0f;
      float post_gain_factor = 1.0f;
    } gain_adjustment;

    bool high_pass_filter_enabled = true;
  } apm;

  // Post-APM AX650 chain.  Keeping these fields in AppConfig lets the C++
  // ALSA backend and the Python orchestration page consume one YAML document.
  struct FastEnhance {
    bool enabled = true;
    std::string model_path = "models-for-asr-chain/ns/fastenhance/fastenhance_48k.axmodel";
    std::string library;
    int npu_core = 0;
    bool monitor_enabled = false;
  } fastenhance;

  struct Vad {
    bool enabled = true;
    std::string model_path = "models-for-asr-chain/vad/ten-vad/ten-vad.axmodel";
    std::string library;
    int npu_core = 0;
    // Python TEN-VAD worker input buffering; accepted here because this
    // shared config is also parsed by run-alsa-apm.
    int input_buffer_ms = 1000;
    int hop_ms = 16;
    float threshold = 0.5f;
    int min_speech_ms = 100;
    // A segment starts checking trailing silence only after this much audio.
    float min_segment_seconds = 3.0f;
    // Safety fallback: force a save when a segment grows this long.
    float max_segment_seconds = 30.0f;
    float trailing_silence_seconds = 3.0f;
  } vad;

  struct Campplus {
    bool enabled = false;
    bool logging_enabled = false;
    // AX650 NPU1 CAMPPlus model.  The deployed image may override this path.
    std::string model_path = "models-for-asr-chain/campplus/campplus.axmodel";
    std::string speaker_table_path = "/tmp/ax-audio-sdk/speakers.json";
    // Embeddings are produced for a 1.5 s window every 0.75 s.
    int min_cluster_size = 4;
    float merge_cosine = 0.8f;
    int min_num_speakers = 1;
    int max_num_speakers = 15;
    float speaker_match_threshold = 0.65f;
  } campplus;

  struct Asr {
    bool enabled = true;
    std::string model_path = "models-for-asr-chain/asr/sensevoice";
    std::string model_type = "sensevoice";
    std::string language = "zh";
    std::string library;
    int npu_core = 1;
    int worker_threads = 1;
  } asr;

  struct Llm {
    bool enabled = false;
    std::string model_path = "models-for-asr-chain/llm/qwen3-1.7b";
    std::string runner = "build/qwen3-worker";
    std::string system_prompt = "你是一个乐于助人的助手。";
    bool translation_mode = false;
    std::string zh_to_en_prompt = "translate the following text into English：";
    std::string en_to_zh_prompt = "以下内容翻译成中文：";
    bool dynamic_load = false;
    int dynamic_load_pool_size = 2;
    bool memory_guard = true;
    int memory_guard_floor_mb = 128;
    int max_tokens = 128;
    float temperature = 0.0f;
    float top_p = 1.0f;
    int top_k = 0;
    float repetition_penalty = 1.0f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    std::string thinking_mode = "disabled";
    bool reset_context = true;
    bool stream_tokens = true;
    int queue_capacity = 0;
    std::string output_file = "/tmp/ax-audio-sdk/llm.txt";
    std::string npu_lock_file = "/tmp/ax-audio-sdk-npu2.lock";
  } llm;

  struct HojoTts {
    bool enabled = false;
    std::string runner = "build/hojo-tts-resident";
    std::string model_path = "models-for-asr-chain/tts/hojo/models";
    // Hojo's model package fixes English voices to 0..12 and Chinese voices
    // to 13..14. Keep separate selections so language routing is unambiguous.
    int english_voice = 9;
    int chinese_voice = 13;
    int max_new_tokens = 1600;
    int queue_capacity = 0;
    bool keep_wav = false;
    std::string output_dir = "/tmp/ax-audio-sdk/hojo-output";
    float playback_gain = 1.0f;
    int npu_core = 1;
  } hojo_tts;

  struct Output {
    std::string temp_dir = "/tmp/ax-audio-sdk/segments";
    std::string text_file = "/tmp/ax-audio-sdk/transcript.txt";
    std::string mp3_dir = "/tmp/ax-audio-sdk/mp3";
    int max_mp3_storage_mb = 512;
    std::string mp3_encoder = "ffmpeg";
  } output;

  struct Runtime {
    std::string input = "alsa";
    std::string input_wav;
    // File input is either paced to the WAV duration or consumed as quickly
    // as downstream processing permits.
    std::string file_input_mode = "realtime";  // "realtime" or "batch".
    std::string apm_binary = "./run-alsa-apm";
    std::string log_level = "info";
    bool playback_enabled = true;
  } runtime;
};

// Loads the complete shared mapping used by audio-pipeline. No external YAML
// library is required on the target. Returns false and fills error on failure.
bool LoadAppConfig(const std::string& path,
                   AppConfig* config,
                   std::string* error);

// Loads only fields consumed by the ALSA/WebRTC APM process. Other sections
// in the shared sdk-config.yaml are intentionally ignored.
bool LoadApmConfig(const std::string& path,
                   AppConfig* config,
                   std::string* error);

bool ValidateAppConfig(const AppConfig& config, std::string* error);
bool ValidateApmConfig(const AppConfig& config, std::string* error);

}  // namespace apm_example

#endif  // EXAMPLES_CONFIGUI_APM_CONFIG_H_
