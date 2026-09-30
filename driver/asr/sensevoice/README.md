# SenseVoice AX650 ASR 适配库

本目录保存 AX650 aarch64 的 `libax_asr_api.so` 和公开 C 头文件。主管线通过
`dlopen` 直接调用该共享库，不再构建或部署独立 runner。

运行模型和 tokenizer 资源位于 `models-for-asr-chain/asr/sensevoice/`。当前生产链路使用
非流式 `sensevoice.axmodel`，并绑定 BIG/NPU2 双核分区。
