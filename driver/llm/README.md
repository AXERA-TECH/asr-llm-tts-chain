# Qwen3 persistent worker

`qwen3-worker` loads Qwen3 once, acknowledges readiness over file descriptor
3, and then consumes length-prefixed UTF-8 prompts from standard input. It
returns framed streaming-token, completion, and error events over descriptor
3. The audio pipeline owns the ASR-to-LLM FIFO and keeps the worker process
alive while that queue is empty.

The vendored AX-LLM runtime is pinned to commit
`c72011486607c43a1637f413d88a9c81ad12bf72`. The worker initializes AX Engine
in BIG_LITTLE mode and uses the BIG/NPU2 partition. Standalone inference is
guarded by the same file lock used around SenseVoice calls. The integrated
audio pipeline passes `--no-lock` because its priority-aware parent arbiter
owns that cross-process lock around the entire Qwen3 request.

The normal SDK build compiles the worker. To build it directly:

```bash
cmake -S driver/llm -B build/qwen3-worker-cmake \
  -DCMAKE_TOOLCHAIN_FILE=driver/llm/toolchains/ax650-aarch64.cmake \
  -DAX650_SDK_DIR=/path/to/board_sdk
cmake --build build/qwen3-worker-cmake --target qwen3-worker -j2
```
