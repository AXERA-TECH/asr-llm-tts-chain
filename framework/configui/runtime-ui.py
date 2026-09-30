#!/usr/bin/env python3
"""Live Gradio dashboard for ASR, Qwen3, and Hojo runtime outputs."""

from __future__ import annotations

import argparse
import re
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import yaml


HERE = Path(__file__).resolve().parent
SDK_DIR = HERE.parents[1]
DEFAULT_CONFIG = SDK_DIR / "config" / "sdk-config.yaml"
LLM_RECORD = re.compile(r"^\[[^\]\r\n]+\] (ASR|Qwen3):(?: |$)")


@dataclass(frozen=True)
class RuntimeOutputs:
    asr_log: Path
    llm_log: Path
    tts_dir: Path


def _nested(config: dict[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"配置缺少字段：{'.'.join(keys)}")
        value = value[key]
    return value


def _resolve_output(path_text: Any, base_dir: Path) -> Path:
    path = Path(str(path_text)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_runtime_outputs(config_path: Path, base_dir: Path = SDK_DIR) -> RuntimeOutputs:
    with config_path.expanduser().open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError("YAML根节点必须是映射")
    return RuntimeOutputs(
        asr_log=_resolve_output(_nested(config, "output", "text_file"), base_dir),
        llm_log=_resolve_output(_nested(config, "llm", "output_file"), base_dir),
        tts_dir=_resolve_output(_nested(config, "hojo_tts", "output_dir"), base_dir),
    )


def _read_history(path: Path, empty_message: str) -> str:
    try:
        if not path.is_file():
            return empty_message
        content = path.read_text(encoding="utf-8", errors="replace")
        return content if content else empty_message
    except OSError as exc:
        return f"读取失败：{path}\n{exc}"


def extract_llm_responses(content: str) -> str:
    """Keep every Qwen3 response while hiding duplicated ASR prompt records."""
    output: list[str] = []
    collecting = False
    found_record = False
    for line in content.splitlines():
        match = LLM_RECORD.match(line)
        if match:
            found_record = True
            collecting = match.group(1) == "Qwen3"
            if collecting:
                output.append(line)
        elif collecting:
            output.append(line)
    if not found_record:
        return content
    return "\n".join(output).rstrip()


def _complete_wav(path: Path) -> bool:
    """Reject a WAV that is still being written or has an invalid RIFF header."""
    try:
        size = path.stat().st_size
        if size < 44:
            return False
        with path.open("rb") as stream:
            header = stream.read(12)
        return (
            header[:4] == b"RIFF"
            and header[8:12] == b"WAVE"
            and int.from_bytes(header[4:8], "little") + 8 <= size
        )
    except OSError:
        return False


def latest_tts_wav(directory: Path) -> str | None:
    try:
        published = [path for path in directory.glob("latest-*.wav") if _complete_wav(path)]
        if published:
            return str(max(published, key=lambda path: (path.stat().st_mtime_ns, path.name)))
        # Compatibility with the first dashboard implementation.
        latest = directory / "latest.wav"
        if _complete_wav(latest):
            return str(latest)
        candidates = [
            path for path in directory.glob("*.wav")
            if path.name != "latest.wav"
            and not path.name.startswith("latest-")
            and _complete_wav(path)
        ]
        if not candidates:
            return None
        return str(max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name)))
    except OSError:
        return None


def read_snapshot(outputs: RuntimeOutputs) -> tuple[str, str, str | None]:
    asr = _read_history(outputs.asr_log, "暂无 SenseVoice 识别结果。")
    llm_content = _read_history(outputs.llm_log, "")
    llm = extract_llm_responses(llm_content) if llm_content else "暂无 Qwen3 响应结果。"
    if not llm:
        llm = "暂无 Qwen3 响应结果。"
    return asr, llm, latest_tts_wav(outputs.tts_dir)


def build_ui(outputs: RuntimeOutputs, refresh_seconds: float = 1.0) -> gr.Blocks:
    if refresh_seconds <= 0:
        raise ValueError("刷新间隔必须大于 0 秒")
    initial = read_snapshot(outputs)
    with gr.Blocks(title="AX650 模型运行状态") as app:
        gr.Markdown(
            "# AX650 模型运行状态\n"
            f"每 {refresh_seconds:g} 秒自动刷新；文字区域保留全部历史，TTS 仅显示最新 WAV。"
        )
        asr_log = gr.Textbox(
            label="ASR（SenseVoice）识别历史",
            value=initial[0], lines=14, max_lines=24,
            interactive=False, autoscroll=True,
        )
        llm_log = gr.Textbox(
            label="LLM（Qwen3）响应历史",
            value=initial[1], lines=14, max_lines=24,
            interactive=False, autoscroll=True,
        )
        tts_audio = gr.Audio(
            label="TTS（Hojo）最新合成音频",
            value=initial[2], type="filepath", format="wav",
            interactive=False, autoplay=False,
        )
        timer = gr.Timer(refresh_seconds)
        timer.tick(
            lambda: read_snapshot(outputs),
            outputs=[asr_log, llm_log, tts_audio],
            concurrency_limit=1,
        )
        app.load(
            lambda: read_snapshot(outputs),
            outputs=[asr_log, llm_log, tts_audio],
        )
    return app


def _port_is_available(host: str, port: int) -> bool:
    if not 1 <= port <= 65535:
        return False
    probe_host = "0.0.0.0" if host in {"0.0.0.0", "::"} else host
    family = socket.AF_INET6 if ":" in probe_host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((probe_host, port))
        except OSError:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-dir", type=Path, default=SDK_DIR)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--refresh-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if not _port_is_available(args.host, args.port):
        raise SystemExit(f"运行状态页面端口不可用：{args.port}")
    outputs = load_runtime_outputs(args.config, args.base_dir)
    outputs.tts_dir.mkdir(parents=True, exist_ok=True)
    build_ui(outputs, args.refresh_seconds).launch(
        server_name=args.host,
        server_port=args.port,
        allowed_paths=[str(outputs.tts_dir)],
        show_error=True,
    )


if __name__ == "__main__":
    main()
