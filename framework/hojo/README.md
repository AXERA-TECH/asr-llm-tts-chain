# Hojo 常驻运行器

`hojo_tts_cpp.cpp` 是完整链路使用的 Hojo 常驻进程源码。启动时一次性加载
LM、`fine_local`、decoder、tokenizer 和文本/音色 embedding，输出
`READY PREPARE_V1` 后接收以下行协议：

```text
PREPARE <id> <voice> <text>
SYNTH <id> <max_new_tokens> "<output.wav>"
CLEAR
```

每个命令以 `OK` 或 `ERR ...` 结束。`PREPARE` 只在 CPU 上分词、构造 prompt
embedding 并缓存在内存中，不调用 NPU；`SYNTH` 消费对应的缓存完成推理，失败也会
移除该缓存；`CLEAR` 清理本次回复的剩余缓存。输出路径支持空格。旧的
`<voice> <max_new_tokens> <output.wav> <text>` 单次合成协议仍可使用。
主管线和 runner 应一起重新构建部署，旧 runner 不支持 `PREPARE_V1` 时启动会报错。

调度以一个 ASR 切片为一个 Cell；开启说话人切分时，每个说话人子切片分别构成 Cell。
同一 Cell 从 ASR 开始到所有 TTS 分段送入播放队列始终持有 NPU2 跨进程锁，
下一 Cell 不能提前执行 ASR。已经送出的音频由 ALSA 独立播放，不必等待播放完毕才
释放 Cell。采集、NPU1 降噪和 VAD 使用各自线程。

```text
Cell n: ASR → Qwen token 输出 → Hojo 分段推理 → 全部分段送入播放队列
                    └─ CPU: 标点切分 → PREPARE（与 Qwen 并行）
                              SYNTH 段 1 → 24→48 kHz → 播放队列
                              SYNTH 段 2 → 24→48 kHz → 播放队列
Cell n+1: ASR → ...
```

当前导出模型的运行接口需要 LM 完成后取得全部音频 token/hidden states，再运行
`fine_local`、decoder 和 ISTFT，尚未提供可直接使用的增量 PCM 解码接口。
主管线先累积到 `hojo_tts.min_chinese_characters` 个非空白、非标点字符（默认 20），
或 `min_english_words` 个英文单词（默认 10），再在后续中英文逗号、句号、问号、
感叹号、分号、冒号等标点处切片。英文缩写中的撇号和小数点不会增加单词数。
无标点文本一直等到当前回复结束，末尾不足阈值的内容也会整段发送，不会强行从半句切开。
每段 WAV 生成后立即转换并送入播放队列，再合成下一段。
这属于分段流式播放，不保证段间无停顿或连续韵律。

Qwen 输出期间，独立准备线程接收完整文本段并发送 `PREPARE`。Qwen 结束后才执行
NPU 上的 Hojo 推理。`llm.stream_tokens` 只控制日志；启用 TTS 后仍会请求 token 帧。
最终响应与流式文本不一致时，丢弃预准备结果，按最终文本重新准备，避免重复/错误
播报。Qwen 失败则丢弃当前回复；`<think>...</think>` 内容不进入合成。

`llm.queue_capacity` 现在控制待处理输入切片的 Cell FIFO；
`hojo_tts.queue_capacity` 控制 CPU 文本准备队列，均以 `0` 表示无界。
非零容量达到上限时会产生背压。`keep_wav` 对每个分段生效，`latest-*.wav` 指向
最新成功生成的分段，便于现有运行页面试听。

`hojo_tts.backlog_threshold` 是独立的积压保护阈值，默认 8，`0` 关闭保护。
统计包含当前待合成/正在合成的子帧和播放队列中尚未播放完的子帧。严格超过阈值时，
清空准备队列和待播放音频，取消当前回复的剩余子帧，播放所选语言/音色的慢说提示。
下一条 LLM 回复恢复正常处理。准备线程中已开始的请求通过代次检查失效，
Qwen 结束后统一清理 runner 缓存，保持 NPU2 推理串行。
`output_dir/queue-status.json` 每 200 ms 原子更新，运行页面显示积压数、阈值和超限次数。

在开发板上预制全部 15 个音色的提示音：

```bash
.venv/bin/python scripts/generate-tts-notices.py
# 只生成当前选定音色，例如：
.venv/bin/python scripts/generate-tts-notices.py --voices 7 13
```

英文音色 `0–12` 生成 “Please speak more slowly, I can’t keep up.”，
中文音色 `13–14` 生成“请慢点说，我跟不上了。”，保存在 `notice_dir/en-<voice>.wav`
和 `notice_dir/zh-<voice>.wav`。生成工具复用 NPU2 锁，已存在的有效 WAV 默认跳过。
缓存缺失时，主管线会在 Qwen 完成后合成并缓存提示音；缓存齐全时超限立即插入播放。

开发机回归测试（使用真实调度/管道协议和模拟模型，不需要 AX650）：

```bash
python framework/hojo/tests/run_cell_tests.py
```

真实模型验证程序位于 `tests/board_bugfix_test.cpp`，会使用现有运行配置，验证
中英文预制提示、音频积压清理、超限后恢复、ASR 过滤和中英翻译/助手输出。
在开发机交叉编译，再将产物复制到开发板 `/tmp`，从 SDK 根目录运行：

```bash
aarch64-linux-gnu-g++ -std=c++17 -O1 -ffunction-sections -fdata-sections \
  -Iframework -Idriver/ns/include -Idriver/vad/include \
  -Idriver/asr/sensevoice/include -Idriver/campplus/include \
  framework/hojo/tests/board_bugfix_test.cpp framework/configui/apm-config.cpp \
  -Wl,--gc-sections -pthread -ldl -o build/board-bugfix-test
# 板端；先停止运行中的主管线，以免重复加载模型：
LD_LIBRARY_PATH=build:/soc/lib:/opt/lib OMP_NUM_THREADS=1 \
  /tmp/asr-bugfix-board-test config/sdk-config.yaml
```

验证程序把输出写入 `/tmp/asr-bugfix-board-validation`，不修改运行配置。

## 音色说明

Hojo 模型共提供 15 个固定音色，索引范围为 `0–14`。英文音色包含女声和
男声；中文音色目前只有女声，没有中文男声。

| 索引 | 音色名称 | 语言 | 性别 |
|---:|---|---|---|
| 0 | `hojo_en_f_01` | 英文 | 女声 |
| 1 | `hojo_en_f_02` | 英文 | 女声 |
| 2 | `hojo_en_f_03` | 英文 | 女声 |
| 3 | `hojo_en_f_04` | 英文 | 女声 |
| 4 | `hojo_en_f_05` | 英文 | 女声 |
| 5 | `hojo_en_f_06` | 英文 | 女声 |
| 6 | `hojo_en_f_07` | 英文 | 女声 |
| 7 | `hojo_en_f_08` | 英文 | 女声 |
| 8 | `hojo_en_m_01` | 英文 | 男声 |
| 9 | `hojo_en_m_02` | 英文 | 男声 |
| 10 | `hojo_en_m_03` | 英文 | 男声 |
| 11 | `hojo_en_m_04` | 英文 | 男声 |
| 12 | `hojo_en_m_05` | 英文 | 男声 |
| 13 | `hojo_zh_f_01` | 中文 | 女声 |
| 14 | `hojo_zh_f_02` | 中文 | 女声 |

主管线通过 `config/sdk-config.yaml` 中的 `hojo_tts.english_voice` 和
`hojo_tts.chinese_voice` 分别配置英文、中文音色。英文音色只允许选择 `0–12`，
中文音色只允许选择 `13–14`。当 Qwen3 回复以 `[zh]` 或 `[en]` 开头时，
主管线按标记选择对应音色；没有语言标记时，主管线的 C++ 代码扫描当前
分段，包含中文字符时选择 `chinese_voice`，否则选择 `english_voice`。
语言判断和音色路由由主管线完成，Hojo 常驻进程只接收已选定的音色索引。

构建产物只有 `build/hojo-tts-resident`。生产部署可携带预制慢说提示 WAV，不包含测试程序、其他试听 WAV、
一次性运行器或开发机上的编译源码。

Hojo 必须使用 `driver/tts/hojo/third_party/ax-llm` 中的固定基线
`dfc4ff38bc187de6e99464bcbf9ef4bfa4c11fb4` 及
`driver/tts/hojo/third_party/patches/ax-llm-core.patch`。Qwen3 使用另一份
ax-llm；二者位于独立进程，可以共存。
