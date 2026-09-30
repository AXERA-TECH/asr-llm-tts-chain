// Hojo-TTS-Light-40M full pipeline (AX650/NPU3): ax-llm LM + fine_local + decoder + ISTFT.
//
// Usage:
//   hojo_tts_cpp <lm_dir> <embeds.bin> <num_tokens> <speech_end_id> <max_new_tokens>
//                <fine_local.axmodel> <decoder.axmodel> <embed_tokens.bin>
//                <speaker_vecs.bin> <voice_idx> <id2code.bin> <out.wav>
//
//   speaker_vecs.bin: float32 [15, 512] (host: voices["speaker_vecs"])
//   id2code.bin:      int64 [17685] (host: token id -> audio code, -1 if not "[N]")
//
// env:
//   AXLLM_DUMP_LOGITS_DIR / AXLLM_DUMP_HIDDEN_DIR  (debug dumps, optional)

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

#include <ax_sys_api.h>
#include <ax_engine_api.h>

#include "runner/LLM.hpp"
#include "runner/ax_model_runner/ax_model_runner_ax650.hpp"
#include "tokenizer.hpp"

#ifdef USE_AXCL
#include "runner/ax_model_runner/ax_model_runner_axcl.hpp"
using ax_runner_t = ax_runner_axcl;
#else
using ax_runner_t = ax_runner_ax650;
#endif

static constexpr int HIDDEN = 512;
static constexpr int SPEAKER_DIM = 192;
static constexpr int TMAX = 2048;
static constexpr int VOCAB = 17685;
static constexpr int NFFT = 1920;
static constexpr int HOP = 480;
static constexpr int SR = 24000;
static constexpr int SPEAKER_SLOTS = 16;
static constexpr int VOICE_COUNT = 15;

static std::vector<unsigned short> load_bin16(const std::string &path);
static std::vector<char> load_bin8(const std::string &path);

static std::string build_prompt(const std::string &text) {
    std::string prompt = "[target_text_start]" + text + "[target_text_end][spk_start]";
    for (int i = 0; i < SPEAKER_SLOTS; ++i)
        prompt += "[spk_emb_" + std::to_string(i) + "]";
    return prompt + "[spk_end][target_speech_start]";
}

static unsigned short float_to_bf16(float value) {
    unsigned int bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<unsigned short>(bits >> 16);
}

static std::vector<unsigned short> make_prompt_embeddings(
        const tokenizer::PreTrainedTokenizer& tokenizer,
        const std::vector<unsigned short>& token_embeddings,
        const std::vector<char>& speaker_embeddings,
        const std::string& text, int voice) {
    const int speaker_start_id = tokenizer.token_to_id("[spk_start]");
    const auto ids = tokenizer.encode(build_prompt(text), false);
    if (speaker_start_id < 0 || ids.empty())
        throw std::runtime_error("invalid Hojo prompt tokens");
    if (token_embeddings.size() % HIDDEN != 0 ||
        speaker_embeddings.size() != VOICE_COUNT * SPEAKER_SLOTS * HIDDEN * sizeof(float) ||
        voice < 0 || voice >= VOICE_COUNT)
        throw std::runtime_error("invalid Hojo embedding assets");
    const size_t vocab = token_embeddings.size() / HIDDEN;
    std::vector<unsigned short> output_embeddings(ids.size() * HIDDEN);
    int speaker_pos = -1;
    for (size_t row = 0; row < ids.size(); ++row) {
        if (ids[row] < 0 || static_cast<size_t>(ids[row]) >= vocab)
            throw std::runtime_error("Hojo token id out of range");
        std::memcpy(output_embeddings.data() + row * HIDDEN,
                    token_embeddings.data() + static_cast<size_t>(ids[row]) * HIDDEN,
                    HIDDEN * sizeof(unsigned short));
        if (speaker_pos < 0 && ids[row] == speaker_start_id)
            speaker_pos = static_cast<int>(row);
    }
    if (speaker_pos < 0 || speaker_pos + SPEAKER_SLOTS >= static_cast<int>(ids.size()))
        throw std::runtime_error("Hojo speaker slots missing");
    const float *speakers = reinterpret_cast<const float *>(speaker_embeddings.data());
    const size_t base = static_cast<size_t>(voice) * SPEAKER_SLOTS * HIDDEN;
    for (int slot = 0; slot < SPEAKER_SLOTS; ++slot)
        for (int col = 0; col < HIDDEN; ++col)
            output_embeddings[(static_cast<size_t>(speaker_pos + 1 + slot) * HIDDEN) + col] =
                float_to_bf16(speakers[base + static_cast<size_t>(slot) * HIDDEN + col]);
    return output_embeddings;
}

static std::vector<unsigned short> load_bin16(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
    {
        std::cerr << "cannot open " << path << std::endl;
        exit(1);
    }
    f.seekg(0, std::ios::end);
    const auto size = f.tellg();
    f.seekg(0, std::ios::beg);
    if (size <= 0 || (size % 2) != 0)
    {
        std::cerr << "bad bin size: " << size << std::endl;
        exit(1);
    }
    std::vector<unsigned short> buf((size_t)size / 2);
    f.read((char *)buf.data(), size);
    return buf;
}

static std::vector<char> load_bin8(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
    {
        std::cerr << "cannot open " << path << std::endl;
        exit(1);
    }
    f.seekg(0, std::ios::end);
    const auto size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf((size_t)size);
    f.read(buf.data(), size);
    return buf;
}

static inline float bf16_to_f32(unsigned short b)
{
    unsigned int u = (unsigned int)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static void rms_norm_rows(float *data, int rows, float eps = 1e-6f)
{
    for (int r = 0; r < rows; ++r)
    {
        float *row = data + (size_t)r * HIDDEN;
        double ss = 0.0;
        for (int i = 0; i < HIDDEN; ++i) ss += (double)row[i] * row[i];
        float inv = 1.0f / std::sqrt((float)(ss / HIDDEN) + eps);
        for (int i = 0; i < HIDDEN; ++i) row[i] *= inv;
    }
}

static void hann(float *w, int n)
{
    for (int i = 0; i < n; ++i) w[i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * i / n));
}

// Mixed-radix Cooley-Tukey FFT (handles 1920 = 2^7 * 3 * 5). sign=+1 inverse.
// tw: precomputed base table e^{sign*2*pi*i*k/1920}, k=0..1919.
static void fft_mixed(std::complex<float> *a, int n,
                      const std::vector<std::complex<float>> &tw, std::complex<float> *ws)
{
    if (n <= 1) return;
    int p = 2;
    while (n % p != 0) ++p;
    const int m = n / p;
    const int base = (int)tw.size();
    const int scale_n = base / n;
    const int scale_p = base / p;
    std::complex<float> *tmp = ws;
    std::complex<float> *out = ws + n;
    for (int i = 0; i < p; ++i)
        for (int j = 0; j < m; ++j)
            tmp[(size_t)i * m + j] = a[(size_t)i + (size_t)p * j];
    for (int i = 0; i < p; ++i)
        fft_mixed(&tmp[(size_t)i * m], m, tw, ws + 2 * n);
    for (int k = 0; k < n; ++k) out[k] = std::complex<float>(0, 0);
    for (int k1 = 0; k1 < m; ++k1)
    {
        const std::complex<float> wn = tw[(size_t)((k1 * scale_n) % base)];
        std::complex<float> w(1.0f, 0.0f);
        for (int i = 0; i < p; ++i)
        {
            const std::complex<float> wpstep = tw[(size_t)((i * scale_p) % base)];
            std::complex<float> wp(1.0f, 0.0f);
            const std::complex<float> gi = tmp[(size_t)i * m + k1] * w;
            for (int k2 = 0; k2 < p; ++k2)
            {
                out[(size_t)k1 + (size_t)m * k2] += gi * wp;
                wp *= wpstep;
            }
            w *= wn;
        }
    }
    for (int k = 0; k < n; ++k) a[k] = out[(size_t)k];
}

static std::vector<float> istft(const float *log_mag, const float *phase, int T, int *out_len)
{
    auto tc0 = std::chrono::steady_clock::now();
    std::vector<float> hann_w(NFFT);
    hann(hann_w.data(), NFFT);
    const int nf = NFFT / 2 + 1;
    const int out_size = (T - 1) * HOP + NFFT;
    std::vector<float> y(out_size, 0.0f);
    std::vector<float> ws(out_size, 0.0f);
    std::vector<float> spec_re(nf), spec_im(nf);
    std::vector<std::complex<float>> full((size_t)NFFT);
    std::vector<std::complex<float>> tw((size_t)NFFT);
    std::vector<std::complex<float>> wsfft(2 * (size_t)NFFT * 2, std::complex<float>(0, 0));
    for (int k = 0; k < NFFT; ++k) tw[k] = std::polar(1.0f, 2.0f * (float)M_PI * (float)k / (float)NFFT);
    auto tc1 = std::chrono::steady_clock::now();
    double t_spec = 0, t_fft = 0, t_ola = 0;
    for (int t = 0; t < T; ++t)
    {
        auto ts0 = std::chrono::steady_clock::now();
        float m, ph;
        for (int f = 0; f < nf; ++f)
        {
            // decoder outputs (nf, TMAX) row-major (all frames, not just T);
            // stride is TMAX, not T. Wrong stride here silently reads garbage
            // and turns the audio into noise.
            m = std::exp(std::min(log_mag[(size_t)f * TMAX + t], std::log(1e2f)));
            ph = phase[(size_t)f * TMAX + t];
            spec_re[f] = m * std::cos(ph);
            spec_im[f] = m * std::sin(ph);
            if (f == NFFT / 2) spec_im[f] = 0.0f;  // Nyquist real-only
        }
        for (int f = 0; f <= NFFT / 2; ++f) full[f] = std::complex<float>(spec_re[f], spec_im[f]);
        for (int f = 1; f < NFFT / 2; ++f) full[(size_t)NFFT - f] = std::conj(full[f]);
        auto ts1 = std::chrono::steady_clock::now();
        fft_mixed(full.data(), NFFT, tw, wsfft.data());
        auto ts2 = std::chrono::steady_clock::now();
        const int off = t * HOP;
        for (int n = 0; n < NFFT; ++n)
        {
            float v = (float)(full[n].real() / NFFT) * hann_w[n];
            y[off + n] += v;
            ws[off + n] += hann_w[n] * hann_w[n];
        }
        auto ts3 = std::chrono::steady_clock::now();
        t_spec += std::chrono::duration<double>(ts1 - ts0).count();
        t_fft += std::chrono::duration<double>(ts2 - ts1).count();
        t_ola += std::chrono::duration<double>(ts3 - ts2).count();
    }
    fprintf(stderr, "ISTFT_PROFILE init=%.3fs spec=%.3fs fft=%.3fs ola=%.3fs\n",
            std::chrono::duration<double>(tc1 - tc0).count(), t_spec, t_fft, t_ola);
    const int pad = (NFFT - HOP) / 2;
    std::vector<float> out(out_size - 2 * pad, 0.0f);
    for (int i = 0; i < out_size - 2 * pad; ++i)
    {
        float e = ws[i + pad];
        out[i] = (e > 1e-6f) ? y[i + pad] / e : 0.0f;
    }
    *out_len = (int)out.size();
    return out;
}

static void write_wav(const std::string &path, const std::vector<float> &x)
{
    std::vector<short> pcm(x.size());
    for (size_t i = 0; i < x.size(); ++i)
    {
        float v = x[i] * 32767.0f;
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        pcm[i] = (short)v;
    }
    FILE *fp = fopen(path.c_str(), "wb");
    if (!fp)
    {
        std::cerr << "cannot write " << path << std::endl;
        exit(1);
    }
    unsigned int data_bytes = (unsigned int)(pcm.size() * 2);
    unsigned int riff = 36 + data_bytes;
    fwrite("RIFF", 1, 4, fp);
    fwrite(&riff, 4, 1, fp);
    fwrite("WAVE", 1, 4, fp);
    fwrite("fmt ", 1, 4, fp);
    unsigned int fmt_size = 16;
    unsigned short fmt_tag = 1, ch = 1;
    unsigned int sr = SR, byte_rate = SR * 2;
    unsigned short block_align = 2, bits = 16;
    fwrite(&fmt_size, 4, 1, fp);
    fwrite(&fmt_tag, 2, 1, fp);
    fwrite(&ch, 2, 1, fp);
    fwrite(&sr, 4, 1, fp);
    fwrite(&byte_rate, 4, 1, fp);
    fwrite(&block_align, 2, 1, fp);
    fwrite(&bits, 2, 1, fp);
    fwrite("data", 1, 4, fp);
    fwrite(&data_bytes, 4, 1, fp);
    fwrite(pcm.data(), 2, pcm.size(), fp);
    fclose(fp);
}

static int run_axmodel(ax_runner_t &runner, const std::vector<std::pair<std::string, std::vector<float>>> &inputs,
                       std::vector<std::pair<std::string, std::vector<float>>> *outputs)
{
    for (const auto &kv : inputs)
    {
        const ax_runner_tensor_t &t = runner.get_input(kv.first);
        if (t.nSize != kv.second.size() * sizeof(float))
        {
            std::cerr << "input " << kv.first << " size mismatch: model " << t.nSize
                      << " vs " << kv.second.size() * sizeof(float) << std::endl;
            return -1;
        }
        memcpy(t.pVirAddr, kv.second.data(), t.nSize);
    }
    runner.set_auto_sync_before_inference(true);
    runner.set_auto_sync_after_inference(true);
    if (runner.inference() != 0) return -1;
    if (outputs)
    {
        for (auto &kv : *outputs)
        {
            const ax_runner_tensor_t &t = runner.get_output(kv.first);
            kv.second.resize(t.nSize / sizeof(float));
            memcpy(kv.second.data(), t.pVirAddr, t.nSize);
        }
    }
    return 0;
}

static int run_once_main(int argc, char **argv)
{
    if (argc < 13)
    {
        std::cerr << "usage: hojo_tts_cpp <lm_dir> <embeds.bin> <num_tokens> <speech_end_id> "
                     "<max_new_tokens> <fine_local.axmodel> <decoder.axmodel> <embed_tokens.bin> "
                     "<speaker_vecs.bin> <voice_idx> <id2code.bin> <out.wav>"
                  << std::endl;
        return 1;
    }
    const std::string model_dir = argv[1];
    const std::string embeds_path = argv[2];
    const int num_tokens = std::atoi(argv[3]);
    const int speech_end_id = std::atoi(argv[4]);
    const int max_new_tokens = std::atoi(argv[5]);
    const std::string fl_path = argv[6];
    const std::string dec_path = argv[7];
    const std::string embed_path = argv[8];
    const std::string spk_path = argv[9];
    const int voice_idx = std::atoi(argv[10]);
    const std::string id2code_path = argv[11];
    const std::string out_wav = argv[12];

    LLMAttrType attr;
    attr.template_filename_axmodel = model_dir + "/qwen3_p8_l%d_together.axmodel";
    attr.filename_post_axmodel = model_dir + "/qwen3_post.axmodel";
    attr.url_tokenizer_model = model_dir + "/tokenizer.txt";
    attr.tokenizer_type = "Qwen3";
    attr.filename_tokens_embed = model_dir + "/embed_tokens.bin";
    attr.tokens_embed_num = VOCAB;
    attr.tokens_embed_size = HIDDEN;
    attr.axmodel_num = 10;
    attr.post_config_path = model_dir + "/post_config.json";
    attr.b_bos = false;
    attr.b_eos = false;

    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    AX_SYS_Init();
    if (AX_ENGINE_Init(&npu_attr) != 0)
    {
        std::cerr << "AX_ENGINE_Init failed" << std::endl;
        return 1;
    }

    LLM llm;
    if (!llm.Init(attr))
    {
        std::cerr << "LLM::Init failed" << std::endl;
        return 1;
    }
    llm.SetExtraStopTokens({speech_end_id});

    auto embeds = load_bin16(embeds_path);
    if ((int)embeds.size() != num_tokens * HIDDEN)
    {
        std::cerr << "embeds size mismatch" << std::endl;
        return 1;
    }

    ax_runner_t fl, dec;
    if (fl.init(fl_path.c_str()) != 0) { std::cerr << "fine_local init failed" << std::endl; return 1; }
    if (dec.init(dec_path.c_str()) != 0) { std::cerr << "decoder init failed" << std::endl; return 1; }

    auto embed_tokens = load_bin16(embed_path);           // [VOCAB, HIDDEN] bf16
    auto spk_bytes = load_bin8(spk_path);                  // [15, HIDDEN] fp32
    auto id2code_bytes = load_bin8(id2code_path);          // [VOCAB] int64
    const float *spk_all = (const float *)spk_bytes.data();
    const long long *id2code = (const long long *)id2code_bytes.data();

    auto t0 = std::chrono::steady_clock::now();
    llm.Run(embeds, max_new_tokens);
    auto t_lm = std::chrono::steady_clock::now();
    auto ids = llm.GetLastGeneratedTokenIds();
    auto hids = llm.GetLastHiddenStates();
    auto t1 = std::chrono::steady_clock::now();
    const double lm_secs = std::chrono::duration<double>(t1 - t0).count();

    // audio tokens: generated ids that map to codes, cut at speech_end
    std::vector<int> audio_gen;  // indices g into ids where id2code valid
    std::vector<long long> codes;
    for (int g = 0; g < (int)ids.size(); ++g)
    {
        if (ids[g] == speech_end_id) break;
        long long c = id2code[ids[g]];
        if (c >= 0) { audio_gen.push_back(g); codes.push_back(c); }
    }
    const int T = (int)audio_gen.size();
    if (T == 0)
    {
        std::cerr << "no audio tokens generated" << std::endl;
        return 1;
    }

    // fine_local inputs
    std::vector<float> hidden_states(TMAX * HIDDEN, 0.0f);
    std::vector<float> coarse_emb(TMAX * HIDDEN, 0.0f);
    std::vector<float> valid(TMAX, 0.0f);
    for (int i = 0; i < T; ++i)
    {
        const int g = audio_gen[i];
        if (g < 0 || g >= (int)hids.size())
        {
            std::cerr << "missing hidden for audio token " << i << std::endl;
            return 1;
        }
        const auto &h = hids[g];
        float *dst = hidden_states.data() + (size_t)i * HIDDEN;
        for (int d = 0; d < HIDDEN; ++d) dst[d] = bf16_to_f32(h[d]);
        const long long tok_id = ids[g];
        const unsigned short *emb = embed_tokens.data() + (size_t)tok_id * HIDDEN;
        float *cdst = coarse_emb.data() + (size_t)i * HIDDEN;
        for (int d = 0; d < HIDDEN; ++d) cdst[d] = bf16_to_f32(emb[d]);
        valid[i] = 1.0f;
    }
    // The reference pipeline feeds RMS-normalized hidden states to fine_local
    // (its LM outputs are per-row RMS=1). axllm returns un-normalized hidden
    // states (per-row RMS ~30-50x larger), so this RMSNorm is REQUIRED.
    rms_norm_rows(hidden_states.data(), T);
    std::vector<float> spk_vec(spk_all + (size_t)voice_idx * SPEAKER_DIM,
                               spk_all + (size_t)(voice_idx + 1) * SPEAKER_DIM);

    std::vector<std::pair<std::string, std::vector<float>>> fl_in = {
        {"hidden_states", hidden_states},
        {"coarse_embeddings", coarse_emb},
        {"speaker_embedding", spk_vec},
        {"valid_mask_fp32", valid},
    };
    std::vector<std::pair<std::string, std::vector<float>>> fl_out = {{"binary_logits", {}}};
    if (run_axmodel(fl, fl_in, &fl_out) != 0) { std::cerr << "fine_local run failed" << std::endl; return 1; }
    auto t_fl = std::chrono::steady_clock::now();
    const float *fl_logits = fl_out[0].second.data();

    std::vector<float> bits(TMAX * 128, 0.0f);
    for (int t = 0; t < T; ++t)
    {
        for (int b = 0; b < 128; ++b)
            bits[(size_t)t * 128 + b] = fl_logits[(size_t)t * 128 + b] > 0.0f ? 1.0f : -1.0f;
    }

    std::vector<std::pair<std::string, std::vector<float>>> dec_in = {{"bits", bits}};
    std::vector<std::pair<std::string, std::vector<float>>> dec_out = {{"mag", {}}, {"phase", {}}};
    if (run_axmodel(dec, dec_in, &dec_out) != 0) { std::cerr << "decoder run failed" << std::endl; return 1; }
    auto t_dec = std::chrono::steady_clock::now();
    const float *mag = dec_out[0].second.data();
    const float *phase = dec_out[1].second.data();

    if (std::getenv("AXLLM_DUMP_DEC_OUT"))
    {
        FILE *fm = fopen("/tmp/cpp_mag.bin", "wb");
        FILE *fp = fopen("/tmp/cpp_phase.bin", "wb");
        if (fm && fp)
        {
            fwrite(mag, sizeof(float), (size_t)(NFFT / 2 + 1) * T, fm);
            fwrite(phase, sizeof(float), (size_t)(NFFT / 2 + 1) * T, fp);
            fclose(fm);
            fclose(fp);
            std::cout << "DUMPED_DEC_OUT T=" << T << std::endl;
        }
    }

    int wav_len = 0;
    std::vector<float> wav = istft(mag, phase, T, &wav_len);
    auto t_istft = std::chrono::steady_clock::now();
    write_wav(out_wav, wav);

    auto t2 = std::chrono::steady_clock::now();
    const double total_secs = std::chrono::duration<double>(t2 - t0).count();
    const double audio_secs = (double)wav_len / SR;
    std::cout << "TTS_OK tokens=" << ids.size() << " audio_tokens=" << T
              << " wav=" << wav_len << " (" << audio_secs << "s) "
              << "lm=" << lm_secs << "s total=" << total_secs << "s "
              << "RTF=" << (total_secs / audio_secs) << std::endl;
    std::cout << "TIME_BREAKDOWN lm=" << std::chrono::duration<double>(t_lm - t0).count()
              << " fl=" << std::chrono::duration<double>(t_fl - t_lm).count()
              << " dec=" << std::chrono::duration<double>(t_dec - t_fl).count()
              << " istft=" << std::chrono::duration<double>(t_istft - t_dec).count()
              << " total=" << total_secs << std::endl;
    return 0;
}

// Resident Hojo runtime.  The original upstream entry point above remains
// available for standalone compatibility; this service keeps all AX handles
// and model buffers alive and only replaces prompt embeddings per request.
class HojoRuntime {
public:
    HojoRuntime(const std::string &model_dir, int speech_end_id,
                const std::string &fl_path, const std::string &dec_path,
                const std::string &embed_path, const std::string &spk_path,
                const std::string &id2code_path)
        : speech_end_id_(speech_end_id),
          model_root_(model_dir.substr(0, model_dir.find_last_of('/'))) {
        // Hojo's exported tokenizer/model intentionally use a numeric vocab and
        // decode-only groups; tell the shared runtime these are expected assets.
        setenv("AXLLM_EXPECT_HOJO_TOKENIZER", "1", 1);
        setenv("AXLLM_EXPECT_DECODE_ONLY_PREFILL", "1", 1);
        LLMAttrType attr;
        attr.template_filename_axmodel = model_dir + "/qwen3_p8_l%d_together.axmodel";
        attr.filename_post_axmodel = model_dir + "/qwen3_post.axmodel";
        attr.url_tokenizer_model = model_dir + "/tokenizer.txt";
        attr.tokenizer_type = "Qwen3";
        attr.filename_tokens_embed = model_dir + "/embed_tokens.bin";
        attr.tokens_embed_num = VOCAB;
        attr.tokens_embed_size = HIDDEN;
        attr.axmodel_num = 10;
        attr.post_config_path = model_dir + "/post_config.json";
        attr.b_bos = false;
        attr.b_eos = false;

        AX_ENGINE_NPU_ATTR_T npu_attr;
        memset(&npu_attr, 0, sizeof(npu_attr));
        npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_BIG_LITTLE;
        if (AX_SYS_Init() != 0 || AX_ENGINE_Init(&npu_attr) != 0)
            throw std::runtime_error("AX Engine initialization failed");
        engine_ready_ = true;
        if (!llm_.Init(attr)) throw std::runtime_error("LLM::Init failed");
        llm_.SetExtraStopTokens({speech_end_id_});
        if (fl_.init(fl_path.c_str()) != 0)
            throw std::runtime_error("fine_local init failed");
        if (dec_.init(dec_path.c_str()) != 0)
            throw std::runtime_error("decoder init failed");
        embed_tokens_ = load_bin16(embed_path);
        tokenizer_ = tokenizer::AutoTokenizer::from_pretrained(model_root_);
        if (!tokenizer_) throw std::runtime_error("failed to load Hojo tokenizer");
        speaker_embeddings_ = load_bin8(model_root_ + "/speaker_embeds.bin");
        spk_bytes_ = load_bin8(spk_path);
        id2code_bytes_ = load_bin8(id2code_path);
        std::cerr << "[hojo] resident models loaded\n";
    }

    ~HojoRuntime() {
        if (engine_ready_) {
            AX_ENGINE_Deinit();
            AX_SYS_Deinit();
        }
    }

    bool Synthesize(std::vector<unsigned short>& embeds,
                    int max_new_tokens, int voice_idx,
                    const std::string &out_wav) {
        if (embeds.empty() || embeds.size() % HIDDEN != 0)
            throw std::runtime_error("embeds size mismatch");
        if (max_new_tokens <= 0)
            throw std::runtime_error("max_new_tokens must be positive");
        if (voice_idx < 0 || voice_idx >= 15)
            throw std::runtime_error("voice index out of range");

        const auto t0 = std::chrono::steady_clock::now();
        llm_.ResetKVCache();
        llm_.Run(embeds, max_new_tokens);
        const auto t_lm = std::chrono::steady_clock::now();
        const auto ids = llm_.GetLastGeneratedTokenIds();
        const auto hids = llm_.GetLastHiddenStates();
        const long long *id2code = reinterpret_cast<const long long *>(id2code_bytes_.data());
        const float *spk_all = reinterpret_cast<const float *>(spk_bytes_.data());
        std::vector<int> audio_gen;
        for (int g = 0; g < (int)ids.size(); ++g) {
            if (ids[g] == speech_end_id_) break;
            if (ids[g] < 0 || ids[g] >= VOCAB) continue;
            if (id2code[ids[g]] >= 0) audio_gen.push_back(g);
        }
        const int T = static_cast<int>(audio_gen.size());
        if (T == 0 || T > TMAX) throw std::runtime_error("invalid audio token count");

        std::vector<float> hidden_states(TMAX * HIDDEN, 0.0f);
        std::vector<float> coarse_emb(TMAX * HIDDEN, 0.0f);
        std::vector<float> valid(TMAX, 0.0f);
        for (int i = 0; i < T; ++i) {
            const int g = audio_gen[i];
            if (g < 0 || g >= (int)hids.size())
                throw std::runtime_error("missing hidden state");
            const auto &h = hids[g];
            float *dst = hidden_states.data() + (size_t)i * HIDDEN;
            for (int d = 0; d < HIDDEN; ++d) dst[d] = bf16_to_f32(h[d]);
            const int tok_id = ids[g];
            const unsigned short *emb = embed_tokens_.data() + (size_t)tok_id * HIDDEN;
            float *cdst = coarse_emb.data() + (size_t)i * HIDDEN;
            for (int d = 0; d < HIDDEN; ++d) cdst[d] = bf16_to_f32(emb[d]);
            valid[i] = 1.0f;
        }
        rms_norm_rows(hidden_states.data(), T);
        std::vector<float> spk_vec(spk_all + (size_t)voice_idx * SPEAKER_DIM,
                                   spk_all + (size_t)(voice_idx + 1) * SPEAKER_DIM);
        std::vector<std::pair<std::string, std::vector<float>>> fl_in = {
            {"hidden_states", hidden_states}, {"coarse_embeddings", coarse_emb},
            {"speaker_embedding", spk_vec}, {"valid_mask_fp32", valid}};
        std::vector<std::pair<std::string, std::vector<float>>> fl_out = {{"binary_logits", {}}};
        if (run_axmodel(fl_, fl_in, &fl_out) != 0)
            throw std::runtime_error("fine_local inference failed");
        const float *fl_logits = fl_out[0].second.data();
        std::vector<float> bits(TMAX * 128, 0.0f);
        for (int t = 0; t < T; ++t)
            for (int b = 0; b < 128; ++b)
                bits[(size_t)t * 128 + b] = fl_logits[(size_t)t * 128 + b] > 0.0f ? 1.0f : -1.0f;
        std::vector<std::pair<std::string, std::vector<float>>> dec_in = {{"bits", bits}};
        std::vector<std::pair<std::string, std::vector<float>>> dec_out = {{"mag", {}}, {"phase", {}}};
        if (run_axmodel(dec_, dec_in, &dec_out) != 0)
            throw std::runtime_error("decoder inference failed");
        int wav_len = 0;
        const auto &mag = dec_out[0].second;
        const auto &phase = dec_out[1].second;
        auto wav = istft(mag.data(), phase.data(), T, &wav_len);
        write_wav(out_wav, wav);
        const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::cerr << "[hojo] resident synthesis tokens=" << ids.size()
                  << " audio_tokens=" << T << " wav=" << wav_len
                  << " lm=" << std::chrono::duration<double>(t_lm - t0).count()
                  << "s total=" << total << "s\n";
        return true;
    }

    // CPU-only: no AX Engine calls, KV cache mutation or model inference.
    // The parent sends PREPARE while Qwen owns the NPU2 Cell.
    void Prepare(uint64_t id, const std::string& text, int voice) {
        if (prepared_.count(id)) throw std::runtime_error("duplicate prepared id");
        auto embeds = make_prompt_embeddings(*tokenizer_, embed_tokens_,
                                             speaker_embeddings_, text, voice);
        prepared_.emplace(id, Prepared{voice, std::move(embeds)});
    }

    bool SynthesizePrepared(uint64_t id, int max_new_tokens,
                            const std::string& output) {
        auto found = prepared_.find(id);
        if (found == prepared_.end()) throw std::runtime_error("unknown prepared id");
        // Consume even on failure; a failed chunk must not leak into another Cell.
        auto prompt = std::move(found->second);
        prepared_.erase(found);
        return Synthesize(prompt.embeds, max_new_tokens, prompt.voice, output);
    }

    void ClearPrepared() { prepared_.clear(); }

    bool SynthesizeText(const std::string &text, int max_new_tokens,
                        int voice_idx, const std::string &out_wav) {
        auto embeds = make_prompt_embeddings(*tokenizer_, embed_tokens_,
                                             speaker_embeddings_, text, voice_idx);
        return Synthesize(embeds, max_new_tokens, voice_idx, out_wav);
    }

private:
    struct Prepared { int voice; std::vector<unsigned short> embeds; };
    std::map<uint64_t, Prepared> prepared_;
    std::shared_ptr<tokenizer::PreTrainedTokenizer> tokenizer_;
    std::vector<char> speaker_embeddings_;
    int speech_end_id_;
    std::string model_root_;
    bool engine_ready_ = false;
    LLM llm_;
    ax_runner_t fl_, dec_;
    std::vector<unsigned short> embed_tokens_;
    std::vector<char> spk_bytes_, id2code_bytes_;
};

int main(int argc, char **argv) {
    if (argc >= 2 && std::string(argv[1]) == "--server") {
        if (argc != 8) {
            std::cerr << "usage: hojo_tts_cpp --server <lm_dir> <speech_end_id> "
                         "<fine_local> <decoder> <embed_tokens> <speaker_vecs> <id2code>\n";
            return 1;
        }
        try {
            const std::string model_root = argv[2];
            HojoRuntime runtime(model_root + "/lm_s8", std::atoi(argv[3]),
                                argv[4], argv[5], model_root + "/lm_s8/embed_tokens.bin",
                                argv[6], argv[7]);
            std::cout << "READY PREPARE_V1" << std::endl;
            std::string line;
            while (std::getline(std::cin, line)) {
                std::istringstream request(line);
                std::string command, output, text;
                int max_new_tokens = 0, voice = 0;
                uint64_t id = 0;
                request >> command;
                try {
                    if (command == "PREPARE") {
                        if (!(request >> id >> voice) || !std::getline(request, text))
                            throw std::runtime_error("invalid PREPARE request");
                        if (!text.empty() && text.front() == ' ') text.erase(0, 1);
                        runtime.Prepare(id, text, voice);
                    } else if (command == "SYNTH") {
                        if (!(request >> id >> max_new_tokens >> std::quoted(output)))
                            throw std::runtime_error("invalid SYNTH request");
                        runtime.SynthesizePrepared(id, max_new_tokens, output);
                    } else if (command == "CLEAR") {
                        runtime.ClearPrepared();
                    } else {
                        // Preserve the original one-shot-per-line server protocol.
                        request.clear();
                        request.str(line);
                        if (!(request >> voice >> max_new_tokens >> output) ||
                            !std::getline(request, text))
                            throw std::runtime_error("invalid request");
                        if (!text.empty() && text.front() == ' ') text.erase(0, 1);
                        runtime.SynthesizeText(text, max_new_tokens, voice, output);
                    }
                    std::cout << "OK" << std::endl;
                } catch (const std::exception &error) {
                    std::cout << "ERR " << error.what() << std::endl;
                }
            }
            return 0;
        } catch (const std::exception &error) {
            std::cerr << "[hojo] resident startup failed: " << error.what() << std::endl;
            return 1;
        }
    }
    return run_once_main(argc, argv);
}
