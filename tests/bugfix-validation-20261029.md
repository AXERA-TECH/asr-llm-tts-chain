# 2026-10-29 修复与验证记录

问题 2、3、4 已修复，并增量部署至 `10.126.29.18:/root/asr-chain`。
问题 1（日语方向误判）按要求暂不处理。测试报告 Wiki 的 HTTPS 连接失败，
本次依据提供的四项问题清单实施。

## 默认参数与行为

| 项目 | 默认配置 | 行为 |
| --- | --- | --- |
| ASR 字符数 | `llm.min_asr_characters: 5` | 去除元数据后按 Unicode 非空白字符计数，必须严格大于阈值 |
| 标点/符号占比 | `llm.max_asr_punctuation_ratio: 0.5` | 必须严格小于阈值；两个条件同时满足才进入 LLM |
| 中文 TTS 切片 | `hojo_tts.min_chinese_characters: 20` | 达到最少字数后在后续标点处分段 |
| 英文 TTS 切片 | `hojo_tts.min_english_words: 10` | 达到最少词数后在后续标点处分段 |
| TTS 积压保护 | `hojo_tts.backlog_threshold: 8` | 待合成和待播放子帧总数严格超过阈值时清理积压并播放慢说提示；0 关闭保护 |

回复末尾不足切片阈值的内容整段发送；无标点长句不强制从半句切开。
超限时丢弃当前回复剩余内容，下一条回复恢复正常。队列状态每 200 ms 原子写入
`hojo_tts.output_dir/queue-status.json`，运行界面自动刷新显示。

助手默认提示词已按要求更新。翻译模式自动使用翻译引擎系统约束，
并在每个请求前附加对应中英文的只输出译文约束。

## 验证结果

- `python3 framework/hojo/tests/run_cell_tests.py`：16 个场景通过。
  覆盖 UTF-8 跨帧、思考标签抑制、最少字/词数、小数与英文缩写、长句完整性、
  纯标点和 ASR 错误过滤、严格阈值边界、翻译提示词、流式/最终文本差异、
  模型错误/断连、文本积压、播放积压、预制提示及缺失缓存回退。
- `.venv/bin/python -m unittest discover -s tests -v`：5 项通过。
  覆盖配置回填/保存、参数校验、队列状态、新旧字段兼容和配置迁移幂等性。
- AX650 交叉编译通过；部署后的依赖检查通过。
- `framework/hojo/tests/board_bugfix_test.cpp` 在开发板使用真实模型验证通过：
  英文音色 7 和中文音色 13 的提示 PCM 与预制 WAV 完全一致，旧 PCM 清除，
  超限后的下一条回复正常合成；纯标点、单字符、50% 标点边界和 ASR 错误均未触发 LLM。
- 真实翻译示例：`Hello, how are you?` → `你好，你怎么样？`；
  `今天天气很好。` → `Today's weather is very good.`，未出现思考标签或解释性前缀。
- 全部 15 个音色的提示音已在板端生成，并同步至本机
  `models-for-asr-chain/tts/hojo/notices`；生成工具重复运行会跳过已有有效 WAV。

## 部署与运行

增量部署复用全部模型，仅传输发生变化的运行文件。板端音频设备、翻译开关、
中英文音色等已有设置保留；新字段自动补齐，旧版出厂系统提示词自动升级。

最终 `build/audio-pipeline` SHA-256：

```text
43a46d6a6ac454677e3e9403d0c7f00b7a57df7a4a180aab2264c8d4f0b2392b
```

回滚备份位于板端 `/root/asr-chain-pre-runtime-fixes-20261029.tar.gz`；
迁移前配置还保存在 `config/sdk-config.yaml.pre-runtime-fixes.bak`。

实际音频处理链已启动，ASR、Qwen3、Hojo、KWS、ALSA 均就绪。
配置页 <http://10.126.29.18:7860> 和运行状态页 <http://10.126.29.18:7862>
均返回 HTTP 200，新参数与队列积压信息显示正常。
首次同时启动多个界面进程时曾出现一次板端 NPU 初始化失败，分开启动后恢复正常。
当前处理链与运行状态页分别运行。

板端日志：`/tmp/asr-bugfix-board-test.log`、`/tmp/asr-bugfix-runtime.log`、
`/tmp/asr-bugfix-config-ui.log`、`/tmp/asr-bugfix-runtime-ui.log`。

默认 ASR 字符数阈值已按要求调整为 5，标点占比阈值保持 50%。
调整后重新通过 16 个处理链场景和 5 项配置测试，覆盖 5 字符拒绝、6 字符通过、
以及字符数满足但标点占比恰好 50% 时拒绝的边界。
