#!/usr/bin/env python3
"""Main Gradio configuration page for the AX650 audio SDK.

The page intentionally follows the stream order.  WebRTC APM has a dedicated
page (``apm-config-ui.py``) so its many controls remain manageable; both pages
read and atomically update the same sdk-config.yaml document.
"""

from __future__ import annotations

import argparse
import os
import socket
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE.parents[1] / "config" / "sdk-config.yaml"
FACTORY_DEFAULT_CONFIG = HERE.parents[1] / "config" / "default-config.yaml"

HOJO_ENGLISH_VOICES = [
    *(f"{index} · hojo_en_f_{index + 1:02d}" for index in range(8)),
    *(f"{index} · hojo_en_m_{index - 7:02d}" for index in range(8, 13)),
]
HOJO_CHINESE_VOICES = ["13 · hojo_zh_f_01", "14 · hojo_zh_f_02"]


class _ConfigDumper(yaml.SafeDumper):
    """Emit multiline strings as escaped scalars understood by the C++ parser."""


def _represent_string(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    style = '"' if "\n" in value or "\r" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_ConfigDumper.add_representer(str, _represent_string)


def _dump(config: dict[str, Any]) -> str:
    return yaml.dump(
        config, Dumper=_ConfigDumper, allow_unicode=True, sort_keys=False
    )


def _read(path: str) -> dict[str, Any]:
    file = Path(path).expanduser()
    if not file.exists():
        return {}
    with file.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError("YAML根节点必须是映射")
    return value


def _save(path: str, config: dict[str, Any]) -> str:
    target = Path(path).expanduser().resolve()
    if target == FACTORY_DEFAULT_CONFIG.resolve():
        raise ValueError("default-config.yaml 是只读默认配置，不能保存覆盖")
    target.parent.mkdir(parents=True, exist_ok=True)
    content = _dump(config)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return content


def _value(config: dict[str, Any], *keys: str, default: Any = "") -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    return value


def _controls_from_config(config: dict[str, Any]) -> tuple[Any, ...]:
    """Return controls in the same order as the main page inputs.

    This is intentionally evaluated on every Gradio page load.  The APM page
    writes the same YAML file while the main page process remains alive, so
    values captured only during ``build_ui`` would become stale.
    """
    return (
        _value(config, "audio", "capture", "device", default="default"),
        _value(config, "audio", "capture", "sample_rate_hz", default=48000),
        _value(config, "audio", "capture", "channels", default=1),
        _value(config, "audio", "playback", "device", default="default"),
        _value(config, "fastenhance", "monitor_enabled", default=False),
        _value(config, "fastenhance", "enabled", default=True),
        _value(config, "fastenhance", "model_path", default="models-for-asr-chain/ns/fastenhance/fastenhance_48k.axmodel"),
        _value(config, "fastenhance", "library", default="build/libfastenhance.so"),
        0,
        _value(config, "vad", "enabled", default=True),
        _value(config, "vad", "library", default="driver/vad/lib/libten_vad.so"),
        _value(config, "vad", "model_path", default="models-for-asr-chain/vad/ten-vad/ten-vad.axmodel"),
        _value(config, "vad", "threshold", default=0.5),
        _value(config, "vad", "min_speech_ms", default=100),
        _value(config, "vad", "min_segment_seconds", default=3),
        _value(config, "vad", "max_segment_seconds", default=30),
        _value(config, "vad", "trailing_silence_seconds", default=3),
        _value(config, "campplus", "enabled", default=False),
        _value(config, "campplus", "logging_enabled", default=False),
        _value(config, "campplus", "model_path", default="models-for-asr-chain/campplus/campplus.axmodel"),
        _value(config, "campplus", "speaker_table_path", default="/tmp/ax-audio-sdk/speakers.json"),
        _value(config, "campplus", "min_cluster_size", default=4),
        _value(config, "campplus", "merge_cosine", default=0.8),
        _value(config, "campplus", "speaker_match_threshold", default=0.65),
        _value(config, "asr", "enabled", default=True),
        _value(config, "asr", "library", default="driver/asr/sensevoice/lib/libax_asr_api.so"),
        _value(config, "asr", "model_path", default="models-for-asr-chain/asr/sensevoice"),
        _value(config, "asr", "model_type", default="sensevoice"),
        _value(config, "asr", "language", default="zh"),
        _value(config, "output", "text_file", default="/tmp/ax-audio-sdk/transcript.txt"),
        _value(config, "output", "temp_dir", default="/tmp/ax-audio-sdk/segments"),
        _value(config, "output", "mp3_dir", default="/tmp/ax-audio-sdk/mp3"),
        _value(config, "output", "max_mp3_storage_mb", default=512),
        _value(config, "llm", "enabled", default=False),
        _value(config, "llm", "model_path", default="models-for-asr-chain/llm/qwen3-1.7b"),
        _value(config, "llm", "runner", default="build/qwen3-worker"),
        _value(config, "llm", "system_prompt", default="你是一个乐于助人的助手。"),
        _value(config, "llm", "translation_mode", default=False),
        _value(config, "llm", "zh_to_en_prompt", default="translate the following text into English："),
        _value(config, "llm", "en_to_zh_prompt", default="以下内容翻译成中文："),
        _value(config, "llm", "dynamic_load", default=False),
        _value(config, "llm", "dynamic_load_pool_size", default=2),
        _value(config, "llm", "memory_guard", default=True),
        _value(config, "llm", "memory_guard_floor_mb", default=128),
        _value(config, "llm", "max_tokens", default=128),
        _value(config, "llm", "temperature", default=0.0),
        _value(config, "llm", "top_p", default=1.0),
        _value(config, "llm", "top_k", default=0),
        _value(config, "llm", "repetition_penalty", default=1.0),
        _value(config, "llm", "frequency_penalty", default=0.0),
        _value(config, "llm", "presence_penalty", default=0.0),
        _value(config, "llm", "thinking_mode", default="disabled"),
        _value(config, "llm", "reset_context", default=True),
        _value(config, "llm", "stream_tokens", default=True),
        _value(config, "llm", "queue_capacity", default=0),
        _value(config, "llm", "output_file", default="/tmp/ax-audio-sdk/llm.txt"),
        _value(config, "llm", "npu_lock_file", default="/tmp/ax-audio-sdk-npu2.lock"),
        _value(config, "hojo_tts", "enabled", default=False),
        _value(config, "hojo_tts", "runner", default="build/hojo-tts-resident"),
        _value(config, "hojo_tts", "model_path", default="models-for-asr-chain/tts/hojo/models"),
        _value(config, "hojo_tts", "english_voice", default=9),
        _value(config, "hojo_tts", "chinese_voice", default=13),
        _value(config, "hojo_tts", "max_new_tokens", default=1600),
        _value(config, "hojo_tts", "queue_capacity", default=0),
        _value(config, "hojo_tts", "keep_wav", default=False),
        _value(config, "hojo_tts", "output_dir", default="/tmp/ax-audio-sdk/hojo-output"),
        _value(config, "hojo_tts", "playback_gain", default=1.0),
        1,
        _value(config, "runtime", "input", default="alsa"),
        _value(config, "runtime", "input_wav", default=""),
        _value(config, "runtime", "file_input_mode", default="realtime"),
    )


def load_values(path_text: str) -> tuple[Any, ...]:
    try:
        config = _read(path_text)
        return (*_controls_from_config(config),
                _dump(config),
                f"已从磁盘加载：{Path(path_text).expanduser().resolve()}")
    except Exception as exc:
        raise gr.Error(f"加载失败：{exc}") from exc


def load_default_values() -> tuple[Any, ...]:
    """Load the immutable factory defaults into the form without saving them."""
    try:
        config = _read(str(FACTORY_DEFAULT_CONFIG))
        if not config:
            raise ValueError(f"默认配置不存在或为空：{FACTORY_DEFAULT_CONFIG}")
        return (
            *_controls_from_config(config),
            _dump(config),
            f"已加载默认配置：{FACTORY_DEFAULT_CONFIG}（尚未保存）",
        )
    except Exception as exc:
        raise gr.Error(f"加载默认配置失败：{exc}") from exc


def _apm_link_javascript(apm_port: int) -> str:
    if not 1 <= apm_port <= 65535:
        raise ValueError("APM 页面端口必须在 1–65535 范围内")
    return f"""
const link = element.querySelector('a');
const target = new URL(window.location.href);
target.port = '{apm_port}';
target.pathname = '/';
target.search = '';
target.hash = '';
link.href = target.toString();
link.title = `打开 ${{target.toString()}}`;
"""


def build_ui(config_path: Path, apm_port: int = 7861) -> gr.Blocks:
    initial = _read(str(config_path))
    with gr.Blocks(title="AX650 音频 SDK 配置") as app:
        gr.Markdown("# AX650 音频处理链配置\nALSA → WebRTC APM → FastEnhance → 低通/16 kHz → TEN-VAD → CampPlus（可选）→ SenseVoice → Qwen3（可选）→ Hojo TTS（可选）")
        path = gr.Textbox(label="统一配置文件", value=str(config_path.resolve()))
        with gr.Tab("ALSA 输入/监听"):
            input_source = gr.Radio(
                [("外设输入", "alsa"), ("文件输入", "file")],
                label="输入源",
                value=_value(initial, "runtime", "input", default="alsa"),
            )
            input_wav = gr.Textbox(
                label="测试音频文件路径（PCM S16_LE WAV）",
                value=_value(initial, "runtime", "input_wav", default=""),
                info="采样率不一致时自动转换到录音采样率；文件输入不会打开录音设备。",
            )
            file_input_mode = gr.Radio(
                [("实时模式", "realtime"), ("批量模式", "batch")],
                label="文件输入模式",
                value=_value(initial, "runtime", "file_input_mode", default="realtime"),
                info="实时模式对齐 WAV 时长；批量模式不等待音频时钟。",
            )
            capture_device = gr.Textbox(label="录音设备", value=_value(initial, "audio", "capture", "device", default="default"))
            capture_rate = gr.Dropdown([8000, 16000, 32000, 48000], label="录音采样率", value=_value(initial, "audio", "capture", "sample_rate_hz", default=48000))
            capture_channels = gr.Number(label="录音声道", value=_value(initial, "audio", "capture", "channels", default=1), precision=0)
            playback_device = gr.Textbox(label="监听/参考放音设备", value=_value(initial, "audio", "playback", "device", default="default"))
            monitor = gr.Checkbox(label="输出 FastEnhance 后音频到扬声器", value=_value(initial, "fastenhance", "monitor_enabled", default=False))
            gr.Markdown("APM 参数保持独立页面，保存时仍写入本文件的 `apm` 字段。")
            gr.HTML(
                '<a href="#" target="_blank" rel="noopener">打开 WebRTC APM 专用设置页面</a>',
                js_on_load=_apm_link_javascript(apm_port),
            )
        with gr.Tab("FastEnhance 降噪"):
            enhance_enabled = gr.Checkbox(label="启用 FastEnhance", value=_value(initial, "fastenhance", "enabled", default=True))
            enhance_model = gr.Textbox(label="模型路径", value=_value(initial, "fastenhance", "model_path", default="models-for-asr-chain/ns/fastenhance/fastenhance_48k.axmodel"))
            enhance_library = gr.Textbox(label="FastEnhance 适配库", value=_value(initial, "fastenhance", "library", default="driver/ns/lib/libfastenhance.so"))
            enhance_core = gr.Number(label="NPU 插槽（固定 0：NPU1 单核分区）", value=0, precision=0, interactive=False)
        with gr.Tab("低通与 TEN-VAD"):
            vad_enabled = gr.Checkbox(label="启用 TEN-VAD", value=_value(initial, "vad", "enabled", default=True))
            vad_library = gr.Textbox(label="TEN-VAD 动态库", value=_value(initial, "vad", "library", default="driver/vad/lib/libten_vad.so"))
            vad_model = gr.Textbox(label="TEN-VAD 模型", value=_value(initial, "vad", "model_path", default="models-for-asr-chain/vad/ten-vad/ten-vad.axmodel"))
            vad_threshold = gr.Slider(0, 1, step=0.01, label="VAD 阈值", value=_value(initial, "vad", "threshold", default=0.5))
            with gr.Row():
                min_speech = gr.Number(label="最短语音 (ms)", value=_value(initial, "vad", "min_speech_ms", default=100), precision=0)
                min_segment = gr.Number(label="最短累积 (s)", value=_value(initial, "vad", "min_segment_seconds", default=3))
                max_segment = gr.Number(label="最长累积兜底 (s)", value=_value(initial, "vad", "max_segment_seconds", default=30))
                trailing = gr.Number(label="末尾静音触发 (s)", value=_value(initial, "vad", "trailing_silence_seconds", default=3))
        with gr.Tab("CampPlus 说话人"):
            campplus_enabled = gr.Checkbox(label="启用 CampPlus 说话人识别", value=_value(initial, "campplus", "enabled", default=False))
            campplus_logging = gr.Checkbox(label="输出 CampPlus 聚类详细日志", value=_value(initial, "campplus", "logging_enabled", default=False))
            campplus_model = gr.Textbox(label="CampPlus AX650 模型", value=_value(initial, "campplus", "model_path", default="models-for-asr-chain/campplus/campplus.axmodel"))
            speaker_table = gr.Textbox(label="长期说话人库（JSON）", value=_value(initial, "campplus", "speaker_table_path", default="/tmp/ax-audio-sdk/speakers.json"))
            with gr.Row():
                min_cluster_size = gr.Number(label="小簇合并/过滤阈值（样本数）", value=_value(initial, "campplus", "min_cluster_size", default=4), precision=0)
                merge_cosine = gr.Slider(-1, 1, step=0.01, label="相似簇合并阈值", value=_value(initial, "campplus", "merge_cosine", default=0.8))
                speaker_match = gr.Slider(-1, 1, step=0.01, label="历史说话人匹配阈值", value=_value(initial, "campplus", "speaker_match_threshold", default=0.65))
        with gr.Tab("SenseVoice / 输出"):
            sensevoice_enabled = gr.Checkbox(label="启用 SenseVoice", value=_value(initial, "asr", "enabled", default=True))
            sensevoice_library = gr.Textbox(label="SenseVoice 动态库", value=_value(initial, "asr", "library", default="driver/asr/sensevoice/lib/libax_asr_api.so"))
            sensevoice_model = gr.Textbox(label="SenseVoice 模型目录", value=_value(initial, "asr", "model_path", default="models-for-asr-chain/asr/sensevoice"))
            sensevoice_type = gr.Textbox(label="模型类型", value=_value(initial, "asr", "model_type", default="sensevoice"))
            language = gr.Textbox(label="默认语言", value=_value(initial, "asr", "language", default="zh"))
            text_file = gr.Textbox(label="文本文件", value=_value(initial, "output", "text_file", default="/tmp/ax-audio-sdk/transcript.txt"))
            temp_dir = gr.Textbox(label="临时 WAV 目录", value=_value(initial, "output", "temp_dir", default="/tmp/ax-audio-sdk/segments"))
            mp3_dir = gr.Textbox(label="MP3 目录", value=_value(initial, "output", "mp3_dir", default="/tmp/ax-audio-sdk/mp3"))
            storage = gr.Number(label="MP3 最大空间 (MB)", value=_value(initial, "output", "max_mp3_storage_mb", default=512))
        with gr.Tab("Qwen3 LLM"):
            llm_enabled = gr.Checkbox(
                label="启用 ASR → Qwen3 总开关",
                value=_value(initial, "llm", "enabled", default=False),
                info="启用后，每个切片按 ASR → Qwen → Hojo 的 Cell 顺序处理；模型保持加载。",
            )
            with gr.Row():
                llm_model = gr.Textbox(label="Qwen3 模型目录", value=_value(initial, "llm", "model_path", default="models-for-asr-chain/llm/qwen3-1.7b"))
                llm_runner = gr.Textbox(label="常驻 Worker", value=_value(initial, "llm", "runner", default="build/qwen3-worker"))
            llm_system = gr.Textbox(label="系统提示词", lines=4, value=_value(initial, "llm", "system_prompt", default="你是一个乐于助人的助手。"))
            llm_translation = gr.Checkbox(
                label="翻译模式",
                value=_value(initial, "llm", "translation_mode", default=False),
                info="仅检查 ASR 裸文本前 10 个字符；中文翻译为英文，其他文本按英文翻译为中文。",
            )
            with gr.Row():
                llm_zh_to_en = gr.Textbox(
                    label="中文 → 英文附加提示词",
                    value=_value(initial, "llm", "zh_to_en_prompt", default="translate the following text into English："),
                )
                llm_en_to_zh = gr.Textbox(
                    label="英文 → 中文附加提示词",
                    value=_value(initial, "llm", "en_to_zh_prompt", default="以下内容翻译成中文："),
                )
            gr.Markdown("#### 模型加载与内存保护")
            with gr.Row():
                llm_dynamic = gr.Checkbox(label="动态加载模型层", value=_value(initial, "llm", "dynamic_load", default=False))
                llm_dynamic_pool = gr.Number(label="动态加载层池大小", value=_value(initial, "llm", "dynamic_load_pool_size", default=2), precision=0)
                llm_memory_guard = gr.Checkbox(label="启用内存保护", value=_value(initial, "llm", "memory_guard", default=True))
                llm_memory_floor = gr.Number(label="保留内存下限 (MB)", value=_value(initial, "llm", "memory_guard_floor_mb", default=128), precision=0)
            gr.Markdown("#### 生成参数")
            with gr.Row():
                llm_max_tokens = gr.Number(label="max_tokens", value=_value(initial, "llm", "max_tokens", default=128), precision=0)
                llm_temperature = gr.Number(label="temperature（0 为贪心）", value=_value(initial, "llm", "temperature", default=0.0))
                llm_top_p = gr.Slider(0.01, 1.0, step=0.01, label="top_p", value=_value(initial, "llm", "top_p", default=1.0))
                llm_top_k = gr.Number(label="top_k（0 为关闭）", value=_value(initial, "llm", "top_k", default=0), precision=0)
            with gr.Row():
                llm_repetition = gr.Number(label="重复惩罚", value=_value(initial, "llm", "repetition_penalty", default=1.0))
                llm_frequency = gr.Slider(-2.0, 2.0, step=0.05, label="频率惩罚", value=_value(initial, "llm", "frequency_penalty", default=0.0))
                llm_presence = gr.Slider(-2.0, 2.0, step=0.05, label="存在惩罚", value=_value(initial, "llm", "presence_penalty", default=0.0))
                llm_thinking = gr.Dropdown(
                    ["disabled", "enabled", "default"], label="思考模式",
                    value=_value(initial, "llm", "thinking_mode", default="disabled"),
                )
            gr.Markdown("#### Cell FIFO、上下文与输出")
            with gr.Row():
                llm_reset = gr.Checkbox(label="每条 ASR 后重置上下文", value=_value(initial, "llm", "reset_context", default=True))
                llm_stream = gr.Checkbox(label="启用流式 token 回调", value=_value(initial, "llm", "stream_tokens", default=True))
                llm_queue = gr.Number(label="待处理 Cell FIFO 容量（0 为无限）", value=_value(initial, "llm", "queue_capacity", default=0), precision=0)
            with gr.Row():
                llm_output = gr.Textbox(label="LLM 输出文件", value=_value(initial, "llm", "output_file", default="/tmp/ax-audio-sdk/llm.txt"))
                llm_lock = gr.Textbox(label="NPU2 跨进程锁", value=_value(initial, "llm", "npu_lock_file", default="/tmp/ax-audio-sdk-npu2.lock"))
        with gr.Tab("Hojo TTS"):
            hojo_enabled = gr.Checkbox(
                label="启用 Hojo TTS 总开关",
                value=_value(initial, "hojo_tts", "enabled", default=False),
                info="Qwen 输出时按标点分段预分词，Qwen 完成后逐段合成并立即送入 ALSA 播放队列。",
            )
            with gr.Row():
                hojo_runner = gr.Textbox(label="Hojo 常驻 runner", value=_value(initial, "hojo_tts", "runner", default="build/hojo-tts-resident"))
                hojo_model = gr.Textbox(label="Hojo 模型目录", value=_value(initial, "hojo_tts", "model_path", default="models-for-asr-chain/tts/hojo/models"))
            with gr.Row():
                hojo_en_voice = gr.Dropdown(
                    choices=[(label, index) for index, label in enumerate(HOJO_ENGLISH_VOICES)],
                    label="英文音色（仅允许 0–12）",
                    value=_value(initial, "hojo_tts", "english_voice", default=9),
                )
                hojo_zh_voice = gr.Dropdown(
                    choices=[(label, index + 13) for index, label in enumerate(HOJO_CHINESE_VOICES)],
                    label="中文音色（仅允许 13–14）",
                    value=_value(initial, "hojo_tts", "chinese_voice", default=13),
                )
            with gr.Row():
                hojo_max_tokens = gr.Number(label="max_new_tokens（最大 2175）", value=_value(initial, "hojo_tts", "max_new_tokens", default=1600), precision=0)
                hojo_queue = gr.Number(label="分段预分词队列容量（0 为无限）", value=_value(initial, "hojo_tts", "queue_capacity", default=0), precision=0)
                hojo_gain = gr.Slider(0.0, 4.0, step=0.05, label="播放增益", value=_value(initial, "hojo_tts", "playback_gain", default=1.0))
                hojo_npu_core = gr.Number(label="NPU 插槽（固定 1：NPU2 双核分区）", value=1, precision=0, interactive=False)
            with gr.Row():
                hojo_keep_wav = gr.Checkbox(label="保留 WAV 文件", value=_value(initial, "hojo_tts", "keep_wav", default=False))
                hojo_output = gr.Textbox(label="WAV 输出目录", value=_value(initial, "hojo_tts", "output_dir", default="/tmp/ax-audio-sdk/hojo-output"))
            gr.Markdown("Hojo 固定输出 24 kHz 单声道 WAV；播放前由主管线转换为 48 kHz。开启 FastEnhance 监听时，两路信号做饱和相加。")
        with gr.Row():
            reload_button = gr.Button("从磁盘重新加载")
            load_default_button = gr.Button("加载默认配置")
            save = gr.Button("校验并保存 sdk-config.yaml", variant="primary")
        preview = gr.Code(label="当前配置", language="yaml", value=_dump(initial))
        status = gr.Textbox(label="状态", interactive=False)

        controls = [capture_device, capture_rate, capture_channels, playback_device, monitor,
                    enhance_enabled, enhance_model, enhance_library,
                    enhance_core,
                    vad_enabled, vad_library, vad_model, vad_threshold, min_speech, min_segment, max_segment, trailing,
                    campplus_enabled, campplus_logging, campplus_model, speaker_table, min_cluster_size, merge_cosine, speaker_match,
                    sensevoice_enabled, sensevoice_library, sensevoice_model, sensevoice_type, language,
                    text_file, temp_dir, mp3_dir, storage,
                    llm_enabled, llm_model, llm_runner, llm_system,
                    llm_translation, llm_zh_to_en, llm_en_to_zh,
                    llm_dynamic, llm_dynamic_pool, llm_memory_guard, llm_memory_floor,
                    llm_max_tokens, llm_temperature, llm_top_p, llm_top_k,
                    llm_repetition, llm_frequency, llm_presence, llm_thinking,
                    llm_reset, llm_stream, llm_queue, llm_output, llm_lock,
                    hojo_enabled, hojo_runner, hojo_model,
                    hojo_en_voice, hojo_zh_voice, hojo_max_tokens,
                    hojo_queue, hojo_keep_wav, hojo_output,
                    hojo_gain, hojo_npu_core,
                    input_source, input_wav, file_input_mode]

        def save_values(path_text: str, *values: Any) -> tuple[str, str]:
            config = _read(path_text)
            audio = config.setdefault("audio", {})
            audio.setdefault("capture", {}).update({"device": str(values[0]).strip(), "sample_rate_hz": int(values[1]), "channels": int(values[2])})
            audio.setdefault("playback", {})["device"] = str(values[3]).strip()
            fastenhance = config.setdefault("fastenhance", {})
            fastenhance.update({
                "enabled": bool(values[5]),
                "model_path": str(values[6]).strip(), "library": str(values[7]).strip(),
                "npu_core": int(values[8]), "monitor_enabled": bool(values[4]),
            })
            vad = config.setdefault("vad", {})
            vad.update({"enabled": bool(values[9]), "library": str(values[10]).strip(), "model_path": str(values[11]).strip(), "threshold": float(values[12]), "min_speech_ms": int(values[13]), "min_segment_seconds": float(values[14]), "max_segment_seconds": float(values[15]), "trailing_silence_seconds": float(values[16])})
            if vad["max_segment_seconds"] <= vad["min_segment_seconds"] + vad["trailing_silence_seconds"]:
                raise gr.Error("最长累积必须大于最短累积 + 末尾静音触发时长")
            config.setdefault("campplus", {}).update({"enabled": bool(values[17]), "logging_enabled": bool(values[18]), "model_path": str(values[19]).strip(), "speaker_table_path": str(values[20]).strip(), "min_cluster_size": int(values[21]), "merge_cosine": float(values[22]), "speaker_match_threshold": float(values[23])})
            asr = config.setdefault("asr", {})
            asr.pop("runner", None)
            asr.update({"enabled": bool(values[24]), "library": str(values[25]).strip(), "model_path": str(values[26]).strip(), "model_type": str(values[27]).strip(), "language": str(values[28]).strip(), "npu_core": 1})
            config.setdefault("output", {}).update({"text_file": str(values[29]).strip(), "temp_dir": str(values[30]).strip(), "mp3_dir": str(values[31]).strip(), "max_mp3_storage_mb": int(values[32])})
            llm = config.setdefault("llm", {})
            llm.update({
                "enabled": bool(values[33]),
                "model_path": str(values[34]).strip(),
                "runner": str(values[35]).strip(),
                "system_prompt": str(values[36]),
                "translation_mode": bool(values[37]),
                "zh_to_en_prompt": str(values[38]),
                "en_to_zh_prompt": str(values[39]),
                "dynamic_load": bool(values[40]),
                "dynamic_load_pool_size": int(values[41]),
                "memory_guard": bool(values[42]),
                "memory_guard_floor_mb": int(values[43]),
                "max_tokens": int(values[44]),
                "temperature": float(values[45]),
                "top_p": float(values[46]),
                "top_k": int(values[47]),
                "repetition_penalty": float(values[48]),
                "frequency_penalty": float(values[49]),
                "presence_penalty": float(values[50]),
                "thinking_mode": str(values[51]),
                "reset_context": bool(values[52]),
                "stream_tokens": bool(values[53]),
                "queue_capacity": int(values[54]),
                "output_file": str(values[55]).strip(),
                "npu_lock_file": str(values[56]).strip(),
            })
            if not llm["model_path"] or not llm["runner"] or not llm["zh_to_en_prompt"] or not llm["en_to_zh_prompt"] or not llm["output_file"] or not llm["npu_lock_file"]:
                raise gr.Error("Qwen3 模型、Worker、翻译提示词、输出文件和 NPU2 锁路径不能为空")
            if llm["dynamic_load_pool_size"] <= 0 or llm["memory_guard_floor_mb"] < 0:
                raise gr.Error("动态加载层池必须大于 0，内存保护下限不能小于 0")
            if not 1 <= llm["max_tokens"] <= 2047 or llm["temperature"] < 0:
                raise gr.Error("max_tokens 必须在 [1, 2047]，temperature 不能小于 0")
            if not 0 < llm["top_p"] <= 1 or llm["top_k"] < 0 or llm["repetition_penalty"] <= 0:
                raise gr.Error("top_p、top_k 或重复惩罚参数无效")
            if not -2 <= llm["frequency_penalty"] <= 2 or not -2 <= llm["presence_penalty"] <= 2:
                raise gr.Error("频率惩罚和存在惩罚必须在 [-2, 2]")
            if llm["thinking_mode"] not in {"default", "enabled", "disabled"} or llm["queue_capacity"] < 0:
                raise gr.Error("思考模式或 FIFO 容量无效")
            hojo = config.setdefault("hojo_tts", {})
            hojo.pop("core", None)
            hojo.pop("keep_prompt", None)
            hojo.update({
                "enabled": bool(values[57]),
                "runner": str(values[58]).strip(),
                "model_path": str(values[59]).strip(),
                "english_voice": int(values[60]),
                "chinese_voice": int(values[61]),
                "max_new_tokens": int(values[62]),
                "queue_capacity": int(values[63]),
                "keep_wav": bool(values[64]),
                "output_dir": str(values[65]).strip(),
                "playback_gain": float(values[66]),
                "npu_core": int(values[67]),
            })
            if not hojo["runner"] or not hojo["model_path"] or not hojo["output_dir"]:
                raise gr.Error("Hojo 常驻程序、模型和输出目录不能为空")
            if not 0 <= hojo["english_voice"] <= 12:
                raise gr.Error("Hojo 英文音色必须在 0–12")
            if not 13 <= hojo["chinese_voice"] <= 14:
                raise gr.Error("Hojo 中文音色必须在 13–14")
            if not 1 <= hojo["max_new_tokens"] <= 2175 or hojo["queue_capacity"] < 0:
                raise gr.Error("Hojo max_new_tokens 必须在 [1, 2175]，FIFO 容量不能小于 0")
            if not 0 <= hojo["playback_gain"] <= 4 or hojo["npu_core"] != 1:
                raise gr.Error("Hojo 播放增益必须在 [0, 4]，NPU 插槽必须为 1")
            if hojo["enabled"] and not llm["enabled"]:
                raise gr.Error("启用 Hojo TTS 前必须启用 Qwen3 LLM")
            runtime = config.setdefault("runtime", {})
            runtime.update({
                "input": str(values[68]),
                "input_wav": str(values[69]).strip(),
                "file_input_mode": str(values[70]),
            })
            if runtime["input"] not in {"alsa", "file"}:
                raise gr.Error("输入源只能选择 alsa 或 file")
            if runtime["file_input_mode"] not in {"realtime", "batch"}:
                raise gr.Error("文件输入模式只能选择 realtime 或 batch")
            if runtime["input"] == "file":
                if not runtime["input_wav"]:
                    raise gr.Error("选择文件输入时必须配置测试音频文件路径")
                # The UI commonly runs on the development host while this
                # path belongs to the AX650 target.  Checking it here would
                # reject valid target paths (and can raise PermissionError for
                # /root).  audio-pipeline and run-alsa-apm perform the strict
                # existence/format checks on the machine that consumes it.
            if runtime["input"] == "alsa" and not config["audio"]["capture"]["device"]:
                raise gr.Error("外设输入模式下，ALSA 录音设备名称不能为空")
            if not config["audio"]["playback"]["device"]:
                raise gr.Error("ALSA 监听/参考放音设备名称不能为空")
            content = _save(path_text, config)
            return content, (
                f"保存成功：{Path(path_text).resolve()}。重启运行时后生效；"
                "文件输入路径将在运行设备上校验。"
            )

        save.click(save_values, inputs=[path, *controls], outputs=[preview, status])
        reload_button.click(load_values, inputs=[path], outputs=[*controls, preview, status])
        load_default_button.click(load_default_values, outputs=[*controls, preview, status])
        # A browser refresh reloads component values through this callback,
        # rather than reusing values captured when the server was started.
        app.load(load_values, inputs=[path], outputs=[*controls, preview, status])
    return app


def _port_is_available(host: str, port: int) -> bool:
    if port == 0:
        return True
    probe_host = "0.0.0.0" if host in {"0.0.0.0", "::"} else host
    family = socket.AF_INET6 if ":" in probe_host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((probe_host, port))
        except OSError:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--apm-port", type=int, default=7861,
        help="WebRTC APM 页面端口，用于生成主页面中的跳转链接",
    )
    args = parser.parse_args()
    if not _port_is_available(args.host, args.port):
        raise SystemExit(
            f"端口 {args.port} 已被占用。请直接打开现有页面，或使用 --port 7870 "
            f"（也可设置 GRADIO_SERVER_PORT）。"
        )
    build_ui(args.config, args.apm_port).launch(
        server_name=args.host, server_port=args.port
    )


if __name__ == "__main__":
    main()
