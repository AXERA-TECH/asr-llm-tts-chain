#include "campplus.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <unordered_map>

#ifdef CAMPPLUS_AVAILABLE
#include "ax_engine_api.h"
#include "ax_sys_api.h"
#include "kaldi-native-fbank/csrc/online-feature.h"
#endif

namespace fs = std::filesystem;

namespace campplus {
namespace {
constexpr std::size_t kRate = 16000;
constexpr std::size_t kWindow = 24000;  // 1.5 s
constexpr std::size_t kStep = 12000;    // 0.75 s
constexpr std::size_t kEmbeddingDim = 192;
constexpr const char* kLogPath = "/tmp/ax-audio-sdk/campplus.log";

void BeginLog(bool enabled, const std::string& segment) {
  if (!enabled) return;
  std::error_code ec;
  fs::create_directories(fs::path(kLogPath).parent_path(), ec);
  std::ofstream out(kLogPath, std::ios::app);
  if (!out) return;
  out << "[campplus][segment="
      << (segment.empty() ? "unknown" : segment)
      << "] ----------------------------------------------->\n";
}

void AppendLog(bool enabled, const std::string& message) {
  if (!enabled) return;
  std::error_code ec;
  fs::create_directories(fs::path(kLogPath).parent_path(), ec);
  std::ofstream out(kLogPath, std::ios::app);
  if (!out) return;
  out << "[campplus] " << message << '\n';
}

float Cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double dot = 0.0, aa = 0.0, bb = 0.0;
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < n; ++i) {
    dot += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i];
  }
  return static_cast<float>(dot / (std::sqrt(aa * bb) + 1e-12));
}

std::vector<float> Mean(const std::vector<std::vector<float>>& e,
                        const std::vector<int>& labels, int label) {
  std::vector<float> out(e.front().size(), 0.0f); int n = 0;
  for (std::size_t i = 0; i < e.size(); ++i) if (labels[i] == label) {
    for (std::size_t j = 0; j < out.size(); ++j) out[j] += e[i][j]; ++n;
  }
  if (n) for (float& x : out) x /= n;
  return out;
}

std::vector<int> UniqueLabels(const std::vector<int>& labels) {
  std::vector<int> unique = labels;
  std::sort(unique.begin(), unique.end());
  unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
  return unique;
}

std::string ClusterSummary(const std::vector<int>& labels) {
  std::ostringstream out;
  bool first_cluster = true;
  for (int id : UniqueLabels(labels)) {
    if (!first_cluster) out << "; ";
    first_cluster = false;
    out << "cluster-" << id << " size="
        << std::count(labels.begin(), labels.end(), id) << " chunks=[";
    bool first_chunk = true;
    for (std::size_t i = 0; i < labels.size(); ++i) if (labels[i] == id) {
      if (!first_chunk) out << ',';
      first_chunk = false;
      out << i;
    }
    out << ']';
  }
  return out.str();
}

std::vector<int16_t> MaskJoins(
    const std::vector<int16_t>& samples, std::size_t window_begin,
    const std::vector<std::size_t>& join_points) {
  std::vector<int16_t> masked = samples;
  if (masked.empty()) return masked;
  const std::size_t window_end = window_begin + masked.size();
  std::vector<std::size_t> covered;
  for (std::size_t point : join_points) {
    if (point > window_begin && point < window_end)
      covered.push_back(point - window_begin);
  }
  if (covered.empty()) return masked;
  if (covered.size() == 1) {
    const std::size_t point = covered.front();
    if (point <= masked.size() / 2)
      std::fill(masked.begin(), masked.begin() + point, 0);
    else
      std::fill(masked.begin() + point, masked.end(), 0);
  } else {
    std::fill(masked.begin(), masked.begin() + covered.front(), 0);
    std::fill(masked.begin() + covered.back(), masked.end(), 0);
  }
  return masked;
}

std::size_t NearestJoin(std::size_t boundary,
                        const std::vector<std::size_t>& join_points) {
  if (join_points.empty()) return boundary;
  return *std::min_element(join_points.begin(), join_points.end(),
      [boundary](std::size_t a, std::size_t b) {
        const std::size_t da = a > boundary ? a - boundary : boundary - a;
        const std::size_t db = b > boundary ? b - boundary : boundary - b;
        return da < db || (da == db && a < b);
      });
}

std::vector<int> Cluster(const std::vector<std::vector<float>>& e,
                         const apm_example::AppConfig::Campplus& c,
                         const std::string& segment) {
  if (e.empty()) return {};
  BeginLog(c.logging_enabled, segment);
  const auto log = [&](const std::string& message) {
    AppendLog(c.logging_enabled, message);
  };
  std::vector<int> labels(e.size()); std::iota(labels.begin(), labels.end(), 0);
  // VAD already supplies speech-only segments, so all utterances use the
  // same agglomerative clustering threshold.
  const float threshold = c.merge_cosine;
  {
    std::ostringstream message;
    message << "clustering begin embeddings=" << e.size()
            << " method=agglomerative"
            << " threshold=" << threshold
            << " min_cluster_size=" << c.min_cluster_size;
    log(message.str());
  }
  while (true) {
    float best = threshold; int bi = -1, bj = -1;
    const std::vector<int> unique = UniqueLabels(labels);
    for (std::size_t i = 0; i < unique.size(); ++i) for (std::size_t j = i + 1; j < unique.size(); ++j) {
      const auto a = Mean(e, labels, unique[i]);
      const auto b = Mean(e, labels, unique[j]);
      const float score = Cosine(a, b);
      if (score >= best) { best = score; bi = unique[i]; bj = unique[j]; }
    }
    if (bi < 0) break;
    {
      std::ostringstream message;
      message << "merge cluster-" << bj << "(size="
              << std::count(labels.begin(), labels.end(), bj) << ") into cluster-"
              << bi << "(size=" << std::count(labels.begin(), labels.end(), bi)
              << ") cosine=" << best;
      log(message.str());
    }
    for (int& x : labels) if (x == bj) x = bi;
  }
  const std::vector<int> unique = UniqueLabels(labels);
  log("candidate clusters (including minor): " + ClusterSummary(labels));
  std::vector<int> major;
  for (int id : unique) {
    if (std::count(labels.begin(), labels.end(), id) > c.min_cluster_size) major.push_back(id);
  }
  if (!major.empty()) for (std::size_t i = 0; i < labels.size(); ++i) {
    if (std::find(major.begin(), major.end(), labels[i]) != major.end()) continue;
    const int minor = labels[i]; int best = major.front(); float score = -2;
    for (int id : major) { const float s = Cosine(e[i], Mean(e, labels, id)); if (s > score) score = s, best = id; }
    {
      std::ostringstream message;
      message << "filter minor cluster-" << minor << " chunk=" << i
              << " -> cluster-" << best << " cosine=" << score;
      log(message.str());
    }
    labels[i] = best;
  } else if (!labels.empty()) {
    const int kept = labels.front();
    log("all clusters are minor; retain cluster-" + std::to_string(kept) +
        " and filter/reassign all others to it");
    std::fill(labels.begin(), labels.end(), kept);
  }
  log("final clusters after filtering: " + ClusterSummary(labels));
  return labels;
}

std::vector<float> CirclePad(const std::vector<float>& x) {
  std::vector<float> y; y.reserve(kWindow);
  if (x.empty()) return std::vector<float>(kWindow, 0.0f);
  while (y.size() < kWindow) y.insert(y.end(), x.begin(), x.begin() + std::min(x.size(), kWindow - y.size()));
  return y;
}

// This compact fallback keeps the source buildable on development hosts.  On
// AX650, CAMPPLUS_AVAILABLE enables the exact kaldi-native-fbank/AX Engine
// implementation below.
#ifdef CAMPPLUS_AVAILABLE
class AxRuntime {
 public:
  AxRuntime() {
    if (AX_SYS_Init() != 0) throw std::runtime_error("AX_SYS_Init failed for CAMPPlus");
    sys_ = true; AX_ENGINE_NPU_ATTR_T attr{};
    if (AX_ENGINE_Init(&attr) != 0) { AX_SYS_Deinit(); sys_ = false; throw std::runtime_error("AX_ENGINE_Init failed for CAMPPlus"); }
    engine_ = true;
  }
  ~AxRuntime() { if (engine_) AX_ENGINE_Deinit(); if (sys_) AX_SYS_Deinit(); }
 private: bool sys_ = false, engine_ = false;
};

class AxModel {
 public:
  explicit AxModel(const std::string& path) { Init(path); }
  ~AxModel() { Release(); }
  std::vector<float> Embed(const std::vector<int16_t>& pcm);
 private:
  void Init(const std::string& path);
  void Release();
  AX_ENGINE_HANDLE handle_ = nullptr; AX_ENGINE_IO_INFO_T* info_ = nullptr;
  AX_ENGINE_IO_T io_{};
};

void AxModel::Init(const std::string& path) {
  std::ifstream f(path, std::ios::binary); if (!f) throw std::runtime_error("cannot open CAMPPlus model: " + path);
  f.seekg(0, std::ios::end); const auto n = f.tellg(); f.seekg(0);
  std::vector<char> model(static_cast<std::size_t>(n)); f.read(model.data(), n);
  if (AX_ENGINE_CreateHandle(&handle_, model.data(), static_cast<AX_U32>(model.size())) != 0 || !handle_ ||
      AX_ENGINE_CreateContext(handle_) != 0 || AX_ENGINE_GetIOInfo(handle_, &info_) != 0) throw std::runtime_error("CAMPPlus AX model init failed");
  io_.nInputSize = info_->nInputSize; io_.nOutputSize = info_->nOutputSize;
  io_.pInputs = new AX_ENGINE_IO_BUFFER_T[io_.nInputSize]{}; io_.pOutputs = new AX_ENGINE_IO_BUFFER_T[io_.nOutputSize]{};
  for (AX_U32 i = 0; i < io_.nInputSize; ++i) { io_.pInputs[i].nSize = info_->pInputs[i].nSize; if (AX_SYS_MemAlloc(&io_.pInputs[i].phyAddr, &io_.pInputs[i].pVirAddr, io_.pInputs[i].nSize, 128, (AX_S8*)"CAMPPLUS-IN") != 0) throw std::runtime_error("CAMPPlus input alloc failed"); }
  for (AX_U32 i = 0; i < io_.nOutputSize; ++i) { io_.pOutputs[i].nSize = info_->pOutputs[i].nSize; if (AX_SYS_MemAlloc(&io_.pOutputs[i].phyAddr, &io_.pOutputs[i].pVirAddr, io_.pOutputs[i].nSize, 128, (AX_S8*)"CAMPPLUS-OUT") != 0) throw std::runtime_error("CAMPPlus output alloc failed"); }
}

void AxModel::Release() {
  for (AX_U32 i = 0; io_.pInputs && i < io_.nInputSize; ++i) if (io_.pInputs[i].pVirAddr) AX_SYS_MemFree(io_.pInputs[i].phyAddr, io_.pInputs[i].pVirAddr);
  for (AX_U32 i = 0; io_.pOutputs && i < io_.nOutputSize; ++i) if (io_.pOutputs[i].pVirAddr) AX_SYS_MemFree(io_.pOutputs[i].phyAddr, io_.pOutputs[i].pVirAddr);
  delete[] io_.pInputs; delete[] io_.pOutputs; io_ = {};
  if (handle_) AX_ENGINE_DestroyHandle(handle_); handle_ = nullptr; info_ = nullptr;
}

std::vector<float> AxModel::Embed(const std::vector<int16_t>& pcm) {
  std::vector<float> wave(pcm.size()); for (std::size_t i = 0; i < pcm.size(); ++i) wave[i] = pcm[i] / 32768.0f;
  wave = CirclePad(wave);
  knf::FbankOptions options; options.frame_opts.samp_freq = kRate; options.frame_opts.dither = 0; options.frame_opts.frame_length_ms = 25; options.frame_opts.frame_shift_ms = 10; options.frame_opts.snip_edges = true; options.frame_opts.window_type = "povey"; options.frame_opts.remove_dc_offset = true; options.frame_opts.round_to_power_of_two = true; options.mel_opts.num_bins = 80; options.mel_opts.low_freq = 20; options.mel_opts.high_freq = 0; options.energy_floor = 0; options.use_energy = false;
  knf::OnlineFbank fbank(options); fbank.AcceptWaveform(kRate, wave.data(), static_cast<int32_t>(wave.size())); fbank.InputFinished();
  std::vector<float> feat(360 * 80); const int n = std::min(360, fbank.NumFramesReady()); std::vector<float> mean(80);
  for (int i = 0; i < n; ++i) for (int j = 0; j < 80; ++j) { feat[i * 80 + j] = fbank.GetFrame(i)[j]; mean[j] += feat[i * 80 + j]; }
  for (float& x : mean) x /= std::max(1, n); for (int i = 0; i < n; ++i) for (int j = 0; j < 80; ++j) feat[i * 80 + j] -= mean[j];
  std::memcpy(io_.pInputs[0].pVirAddr, feat.data(), feat.size() * sizeof(float)); AX_SYS_MflushCache(io_.pInputs[0].phyAddr, io_.pInputs[0].pVirAddr, io_.pInputs[0].nSize);
  if (AX_ENGINE_RunSync(handle_, &io_) != 0) throw std::runtime_error("CAMPPlus inference failed");
  AX_SYS_MinvalidateCache(io_.pOutputs[0].phyAddr, io_.pOutputs[0].pVirAddr, io_.pOutputs[0].nSize);
  std::vector<float> out(kEmbeddingDim); std::memcpy(out.data(), io_.pOutputs[0].pVirAddr, out.size() * sizeof(float)); return out;
}
#endif

struct Speaker { std::string id; std::vector<float> center; int count = 0; std::int64_t last_seen = 0; };

void LoadTable(const std::string& path, std::vector<Speaker>* table, int* next) {
  std::ifstream in(path); if (!in) return; std::string line;
  while (std::getline(in, line)) {
    const auto p = line.find("\"speaker_id\""); if (p == std::string::npos) continue;
    const auto q = line.find('"', line.find(':', p) + 1); const auto r = line.find('"', q + 1); if (q == std::string::npos || r == std::string::npos) continue;
    Speaker s; s.id = line.substr(q + 1, r - q - 1);
    const auto c = line.find("\"sample_count\""); if (c != std::string::npos) s.count = std::atoi(line.c_str() + line.find(':', c) + 1);
    const auto seen = line.find("\"last_seen_time\""); if (seen != std::string::npos) s.last_seen = std::atoll(line.c_str() + line.find(':', seen) + 1);
    const auto a = line.find('[', line.find("\"embedding_center\"")); const auto b = line.find(']', a);
    if (a != std::string::npos && b != std::string::npos) { std::stringstream values(line.substr(a + 1, b - a - 1)); std::string token; while (std::getline(values, token, ',')) { try { s.center.push_back(std::stof(token)); } catch (...) {} } }
    table->push_back(std::move(s));
  }
  for (const auto& s : *table) { const auto p = s.id.find_last_of('-'); if (p != std::string::npos) *next = std::max(*next, std::atoi(s.id.c_str() + p + 1) + 1); }
}

void SaveTable(const std::string& path, const std::vector<Speaker>& table) {
  fs::path p(path); if (!p.parent_path().empty()) fs::create_directories(p.parent_path()); const fs::path tmp = p.string() + ".tmp"; std::ofstream out(tmp); out << "{\n  \"speakers\": [\n";
  for (std::size_t i = 0; i < table.size(); ++i) { const auto& s = table[i]; out << "    {\"speaker_id\":\"" << s.id << "\",\"sample_count\":" << s.count << ",\"last_seen_time\":" << s.last_seen << ",\"embedding_center\":["; for (std::size_t j = 0; j < s.center.size(); ++j) out << (j ? "," : "") << std::setprecision(8) << s.center[j]; out << "]}" << (i + 1 == table.size() ? "\n" : ",\n"); }
  out << "  ]\n}\n"; out.close(); std::error_code ec; fs::rename(tmp, p, ec); if (ec) fs::remove(tmp);
}
}  // namespace

#ifdef CAMPPLUS_CLUSTER_TEST_API
std::vector<int> ClusterForTest(
    const std::vector<std::vector<float>>& embeddings,
    const apm_example::AppConfig::Campplus& config,
    const std::string& segment_name) {
  return Cluster(embeddings, config, segment_name);
}

std::vector<int16_t> MaskJoinsForTest(
    const std::vector<int16_t>& samples, std::size_t window_begin,
    const std::vector<std::size_t>& join_points) {
  return MaskJoins(samples, window_begin, join_points);
}

std::size_t NearestJoinForTest(
    std::size_t boundary, const std::vector<std::size_t>& join_points) {
  return NearestJoin(boundary, join_points);
}
#endif

struct Session::Impl {
  explicit Impl(const apm_example::AppConfig::Campplus& c, std::function<void()> l, std::function<void()> u) : config(c), lock(std::move(l)), unlock(std::move(u)) {
    if (!config.enabled) return;
    LoadTable(config.speaker_table_path, &speakers, &next_id);
#ifdef CAMPPLUS_AVAILABLE
    try { runtime = std::make_unique<AxRuntime>(); model = std::make_unique<AxModel>(config.model_path); ready = true; } catch (const std::exception& e) { std::fprintf(stderr, "[campplus] %s\n", e.what()); }
#else
    std::fprintf(stderr, "[campplus] built without AX650 CAMPPlus support\n");
#endif
  }
  apm_example::AppConfig::Campplus config; bool ready = false; std::function<void()> lock, unlock; std::vector<int16_t> samples; std::vector<std::vector<float>> embeddings; std::vector<std::pair<std::size_t, std::size_t>> times; std::vector<std::size_t> join_points; std::vector<Speaker> speakers; int next_id = 1;
#ifdef CAMPPLUS_AVAILABLE
  std::unique_ptr<AxRuntime> runtime; std::unique_ptr<AxModel> model;
  std::vector<float> Embed(const std::vector<int16_t>& pcm) {
    if (lock) lock();
    try { auto result = model->Embed(pcm); if (unlock) unlock(); return result; }
    catch (...) { if (unlock) unlock(); throw; }
  }
#endif
};

Session::Session(const apm_example::AppConfig::Campplus& c, std::function<void()> l, std::function<void()> u) : impl_(new Impl(c, std::move(l), std::move(u))) {}
Session::~Session() { delete impl_; }
bool Session::available() const { return impl_->ready; }
void Session::Reset() { impl_->samples.clear(); impl_->embeddings.clear(); impl_->times.clear(); impl_->join_points.clear(); }
void Session::MarkJoinPoint(std::size_t sample_offset) {
  if (!impl_->ready || sample_offset == 0) return;
  if (impl_->join_points.empty() || impl_->join_points.back() != sample_offset)
    impl_->join_points.push_back(sample_offset);
}
void Session::Push(const std::vector<int16_t>& x) {
  if (!impl_->ready) return; impl_->samples.insert(impl_->samples.end(), x.begin(), x.end());
#ifdef CAMPPLUS_AVAILABLE
  while (impl_->samples.size() >= kWindow + impl_->embeddings.size() * kStep) {
    const std::size_t begin = impl_->embeddings.size() * kStep; std::vector<int16_t> chunk(impl_->samples.begin() + begin, impl_->samples.begin() + begin + kWindow); chunk = MaskJoins(chunk, begin, impl_->join_points); impl_->embeddings.push_back(impl_->Embed(chunk)); impl_->times.push_back({begin, begin + kWindow});
  }
#endif
}

std::vector<LabeledPart> Session::Finalize(std::size_t sample_count,
                                           const std::string& segment_name,
                                           const std::vector<std::size_t>& join_points) {
  std::vector<LabeledPart> out; if (!impl_->ready || impl_->samples.empty()) { Reset(); return out; }
#ifdef CAMPPLUS_AVAILABLE
  const std::string segment = segment_name.empty() ? "unknown" : segment_name;
  const auto log = [&](const std::string& message) {
    AppendLog(impl_->config.logging_enabled, message);
  };
  if (!join_points.empty()) impl_->join_points = join_points;
  if (impl_->embeddings.empty()) { impl_->embeddings.push_back(impl_->Embed(MaskJoins(impl_->samples, 0, impl_->join_points))); impl_->times.push_back({0, impl_->samples.size()}); }
  const auto labels = Cluster(impl_->embeddings, impl_->config, segment); std::unordered_map<int, std::string> ids; const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  for (int label : labels) if (!ids.count(label)) {
    const auto center = Mean(impl_->embeddings, labels, label);
    const int cluster_count = std::max(1, static_cast<int>(std::count(labels.begin(), labels.end(), label)));
    int best = -1; float score = impl_->config.speaker_match_threshold;
    for (std::size_t i = 0; i < impl_->speakers.size(); ++i) { const float s = Cosine(center, impl_->speakers[i].center); if (s >= score) score = s, best = static_cast<int>(i); }
    if (best < 0) {
      Speaker s; s.id = "speaker-" + std::to_string(impl_->next_id++); s.center = center; s.count = cluster_count; s.last_seen = now; impl_->speakers.push_back(std::move(s)); best = static_cast<int>(impl_->speakers.size() - 1);
      std::ostringstream message; message << "cluster-" << label
          << " created new " << impl_->speakers[best].id
          << " cluster_size=" << cluster_count
          << " (no history cosine >= " << impl_->config.speaker_match_threshold
          << ')'; log(message.str());
    } else {
      const std::string matched_id = impl_->speakers[best].id;
      const int previous_count = impl_->speakers[best].count;
      auto& s = impl_->speakers[best]; const int n = std::max(1, s.count); if (s.center.size() != center.size()) s.center = center; else for (std::size_t j = 0; j < center.size(); ++j) s.center[j] = (s.center[j] * n + center[j] * cluster_count) / (n + cluster_count); s.count = n + cluster_count; s.last_seen = now;
      std::ostringstream message; message << "cluster-" << label
          << " matched existing " << matched_id << " cosine=" << score
          << " cluster_size=" << cluster_count << " history_count="
          << previous_count << "->" << s.count; log(message.str());
    }
    ids[label] = impl_->speakers[best].id;
  }
  std::size_t part_begin = 0;
  std::string current_id = ids[labels.front()];
  for (std::size_t i = 1; i < labels.size(); ++i) {
    const std::string next_id = ids[labels[i]];
    if (next_id == current_id) continue;
    const std::size_t nominal = impl_->times[i].first;
    const std::size_t boundary = std::min(sample_count,
        std::max(part_begin, NearestJoin(nominal, impl_->join_points)));
    if (boundary > part_begin) out.push_back({part_begin, boundary, current_id});
    part_begin = boundary;
    current_id = next_id;
  }
  if (sample_count > part_begin) out.push_back({part_begin, sample_count, current_id});
  SaveTable(impl_->config.speaker_table_path, impl_->speakers);
  {
    std::ostringstream message; message << "clustering complete final_clusters="
        << UniqueLabels(labels).size() << " speaker_table_size="
        << impl_->speakers.size() << " labeled_parts=" << out.size();
    log(message.str());
  }
#endif
  Reset(); return out;
}
}  // namespace campplus
