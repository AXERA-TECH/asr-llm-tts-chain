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
因此采用中英文逗号、句号、问号、感叹号、分号、冒号等标点分段合成，每段 WAV
生成后立即转换并送入播放队列，再合成下一段。无标点文本最多积累 100 个 Unicode
字符，英文在 80 字符后优先按空白断开；小数点不会拆开数字，末尾无标点文本也会发送。
这属于分段流式播放，不保证段间无停顿或连续韵律。

Qwen 输出期间，独立准备线程接收完整文本段并发送 `PREPARE`。Qwen 结束后才执行
NPU 上的 Hojo 推理。`llm.stream_tokens` 只控制日志；启用 TTS 后仍会请求 token 帧。
最终响应与流式文本不一致时，丢弃预准备结果，按最终文本重新准备，避免重复/错误
播报。Qwen 失败则丢弃当前回复；`<think>...</think>` 内容不进入合成。

`llm.queue_capacity` 现在控制待处理输入切片的 Cell FIFO；
`hojo_tts.queue_capacity` 控制 CPU 文本准备队列，均以 `0` 表示无界。
非零容量达到上限时会产生背压。`keep_wav` 对每个分段生效，`latest-*.wav` 指向
最新成功生成的分段，便于现有运行页面试听。

开发机回归测试（使用真实调度/管道协议和模拟模型，不需要 AX650）：

```bash
python framework/hojo/tests/run_cell_tests.py
```

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

构建产物只有 `build/hojo-tts-resident`。生产部署不包含测试程序、试听 WAV、
一次性运行器或开发机上的编译源码。

Hojo 必须使用 `driver/tts/hojo/third_party/ax-llm` 中的固定基线
`dfc4ff38bc187de6e99464bcbf9ef4bfa4c11fb4` 及
`driver/tts/hojo/third_party/patches/ax-llm-core.patch`。Qwen3 使用另一份
ax-llm；二者位于独立进程，可以共存。
