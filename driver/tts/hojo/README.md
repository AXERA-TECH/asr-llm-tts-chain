# Hojo TTS 构建目录

该目录只构建完整链路使用的 `hojo-tts-resident`。它静态链接 Hojo 专用的
`hojo_axllm_runtime` 和 `third_party/tokenizer.cpp`。

Hojo ax-llm 固定要求：

```text
源码：third_party/ax-llm
提交：dfc4ff38bc187de6e99464bcbf9ef4bfa4c11fb4
补丁：third_party/patches/ax-llm-core.patch
```

Qwen3 使用 `driver/llm/third_party/ax-llm`，两套运行时位于不同可执行进程中。

单独构建：

```bash
cmake -S driver/tts/hojo -B build/hojo-tts-cmake \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/driver/tts/hojo/toolchains/ax650-aarch64.cmake" \
  -DAX650_SDK_DIR=/path/to/ax650-board-sdk \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$PWD/build"
cmake --build build/hojo-tts-cmake --target hojo-tts-resident -j2
```
