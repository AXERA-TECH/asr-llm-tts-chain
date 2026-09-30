# 配置页面

SDK 使用一个配置文件：`../../config/sdk-config.yaml`。

`runtime-ui.py` 是随 `scripts/run-target.sh` 自动启动的运行状态页面，默认端口为
`7862`。它从配置中读取 ASR、LLM 和 Hojo 输出路径：前两个文本框显示全部历史，
TTS 播放器只显示 `hojo_tts.output_dir/latest-*.wav` 中最近一次合成结果。
主管线退出时该页面也会由启动脚本停止。

启动两个页面：

```bash
cd /path/to/ax-audio-sdk
./start-ui.sh
```

主页面默认是 `http://127.0.0.1:7860`，包含 ALSA、FastEnhance 监听/AEC
参考、低通/抽取、TEN-VAD、SenseVoice、Qwen3、Hojo TTS 和输出存储配置；APM 页面是
`http://127.0.0.1:7861`。
主页面中的“打开 WebRTC APM 专用设置页面”链接会自动沿用浏览器当前访问的
协议和主机名，只将端口切换为 APM 端口；因此使用 `127.0.0.1`、`localhost`
或局域网 IP 访问主页面时，都会跳转到同一主机的 APM 页面。
“ALSA 输入/监听”页可在外设输入和 WAV 文件输入之间切换，并设置文件输入的实时或批量
模式。配置页面可能运行在开发机，因此保存时只校验文件路径非空；目标板启动处理链时会
严格检查目标板上的文件是否存在，并校验 WAV 格式和采样率。
WAV 采样率与配置的录音采样率不同时，APM 前端会自动进行高质量 sinc 重采样。
APM 页面只编辑 `audio`、`buffers`、`apm` 字段，保存时保留其余模型和输出配置。
两个页面都可用“加载默认配置”将只读的 `config/default-config.yaml` 回填到表单；
该操作不会立即覆盖运行配置，检查后点击保存才会写入 `config/sdk-config.yaml`。

也可以分别启动：

```bash
python3 framework/configui/sdk-config-ui.py --config config/sdk-config.yaml --port 7860 --apm-port 7861
python3 framework/configui/apm-config-ui.py --config config/sdk-config.yaml --port 7861
```

页面保存采用临时文件、`fsync` 和原子替换。运行中的处理链不热加载，保存后
重启 `build/audio-pipeline` 或板端服务生效。Hojo 页会分别限制英文音色为
`0–12`、中文音色为 `13–14`。

如果提示端口占用，说明对应页面已经启动或端口被其他程序使用。可直接访问
现有页面，或者指定新端口，例如：

```bash
uv run python framework/configui/apm-config-ui.py --config config/sdk-config.yaml --port 7871
MAIN_PORT=7870 APM_PORT=7871 ./scripts/start-ui.sh
```
