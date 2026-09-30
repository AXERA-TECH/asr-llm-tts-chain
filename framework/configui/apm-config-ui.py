#!/usr/bin/env python3
"""Standalone WebRTC APM editor backed by the unified sdk-config.yaml."""

from __future__ import annotations

import argparse
import math
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
NATIVE_RATES = [8000, 16000, 32000, 48000]

DEFAULTS: dict[str, Any] = {
    "audio": {
        "playback": {"device": "hw:1,3", "sample_rate_hz": 48000, "channels": 2},
        "capture": {"device": "hw:1,2", "sample_rate_hz": 48000, "channels": 2},
    },
    "buffers": {"period_ms": 10, "alsa_buffer_ms": 40, "application_buffer_ms": 1000},
    "apm": {
        "aec": {
            "enabled": True,
            "mode": "aec3",
            "multi_channel_render": True,
            "multi_channel_capture": True,
        },
        "delay": {"base_ms": 0},
        "ns": {"enabled": True, "level": "high"},
        "agc1": {
            "enabled": False,
            "mode": "fixed_digital",
            "target_level_dbfs": 3,
            "compression_gain_db": 9,
            "limiter": True,
        },
        "agc2": {
            "enabled": True,
            "fixed_gain_db": 0.0,
            "adaptive_digital": False,
            "adaptive_headroom_db": 5.0,
            "adaptive_max_gain_db": 50.0,
            "adaptive_initial_gain_db": 15.0,
            "adaptive_max_gain_change_db_per_second": 6.0,
            "adaptive_max_output_noise_level_dbfs": -50.0,
        },
        "gain_adjustment": {
            "enabled": True,
            "pre_gain_factor": 1.0,
            "post_gain_factor": 1.0,
        },
        "high_pass_filter_enabled": True,
    },
}


def nested(config: dict[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        value = value[key]
    return value


def db_to_linear(db: float) -> float:
    """Convert amplitude gain in dB to a linear multiplier."""
    return round(10.0 ** (float(db) / 20.0), 8)


def linear_to_db(factor: float) -> float:
    """Convert a positive linear amplitude multiplier to dB."""
    factor = float(factor)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("前置/后置线性增益必须是大于0的有限数")
    return round(20.0 * math.log10(factor), 4)


def validate(config: dict[str, Any]) -> None:
    playback = config["audio"]["playback"]
    capture = config["audio"]["capture"]
    buffers = config["buffers"]
    apm = config["apm"]

    if playback["sample_rate_hz"] not in NATIVE_RATES or capture["sample_rate_hz"] not in NATIVE_RATES:
        raise ValueError("S16 APM采样率只能是 8000/16000/32000/48000 Hz")
    if playback["sample_rate_hz"] != capture["sample_rate_hz"]:
        raise ValueError("当前S16 APM接口要求放音参考速率与录音速率相同")
    if not 1 <= playback["channels"] <= 8 or not 1 <= capture["channels"] <= 8:
        raise ValueError("放音和录音声道数必须在1到8之间")
    if not playback["device"].strip() or not capture["device"].strip():
        raise ValueError("ALSA设备名称不能为空")
    if buffers["period_ms"] <= 0:
        raise ValueError("ALSA period必须大于0 ms")
    if buffers["alsa_buffer_ms"] < buffers["period_ms"]:
        raise ValueError("ALSA buffer不能小于period")
    if buffers["application_buffer_ms"] <= 0:
        raise ValueError("应用队列深度必须大于0 ms")
    if not 0 <= apm["delay"]["base_ms"] <= 500:
        raise ValueError("基础延迟必须在0到500 ms之间")
    if not 0 <= apm["agc1"]["target_level_dbfs"] <= 31:
        raise ValueError("AGC1目标电平必须在0到31 dBFS之间")
    if not 0 <= apm["agc1"]["compression_gain_db"] <= 90:
        raise ValueError("AGC1压缩增益必须在0到90 dB之间")
    agc2 = apm["agc2"]
    if not 0 <= agc2["fixed_gain_db"] < 50:
        raise ValueError("AGC2固定数字增益必须大于等于0且小于50 dB")
    if agc2["adaptive_headroom_db"] < 0 or agc2["adaptive_initial_gain_db"] < 0 or agc2["adaptive_max_gain_db"] <= 0:
        raise ValueError("AGC2自适应余量/初始增益不能为负，最大增益必须大于0")
    if agc2["adaptive_max_gain_change_db_per_second"] <= 0:
        raise ValueError("AGC2最大增益变化速度必须大于0")
    if agc2["adaptive_max_output_noise_level_dbfs"] > 0:
        raise ValueError("AGC2最大输出噪声电平必须小于等于0 dBFS")
    adjustment = apm["gain_adjustment"]
    for label, factor in (
        ("前置", adjustment["pre_gain_factor"]),
        ("后置", adjustment["post_gain_factor"]),
    ):
        if not math.isfinite(float(factor)) or not 0 < float(factor) <= 1000:
            raise ValueError(f"{label}线性增益必须在(0, 1000]范围内")


def make_config(*values: Any) -> dict[str, Any]:
    (
        playback_device, playback_rate, playback_channels,
        capture_device, capture_rate, capture_channels,
        period_ms, alsa_buffer_ms, application_buffer_ms,
        aec_enabled, aec_mode, multi_render, multi_capture, base_delay_ms,
        ns_enabled, ns_level,
        agc1_enabled, agc1_mode, agc1_target, agc1_gain, agc1_limiter,
        agc2_enabled, agc2_fixed_gain, agc2_adaptive, agc2_headroom,
        agc2_max_gain, agc2_initial_gain, agc2_change_rate, agc2_noise_floor,
        gain_adjustment_enabled, pre_gain_db, post_gain_db,
        high_pass_filter_enabled,
    ) = values
    config = {
        "audio": {
            "playback": {
                "device": str(playback_device).strip(),
                "sample_rate_hz": int(playback_rate),
                "channels": int(playback_channels),
            },
            "capture": {
                "device": str(capture_device).strip(),
                "sample_rate_hz": int(capture_rate),
                "channels": int(capture_channels),
            },
        },
        "buffers": {
            "period_ms": int(period_ms),
            "alsa_buffer_ms": int(alsa_buffer_ms),
            "application_buffer_ms": int(application_buffer_ms),
        },
        "apm": {
            "aec": {
                "enabled": bool(aec_enabled),
                "mode": str(aec_mode),
                "multi_channel_render": bool(multi_render),
                "multi_channel_capture": bool(multi_capture),
            },
            "delay": {"base_ms": int(base_delay_ms)},
            "ns": {"enabled": bool(ns_enabled), "level": str(ns_level)},
            "agc1": {
                "enabled": bool(agc1_enabled),
                "mode": str(agc1_mode),
                "target_level_dbfs": int(agc1_target),
                "compression_gain_db": int(agc1_gain),
                "limiter": bool(agc1_limiter),
            },
            "agc2": {
                "enabled": bool(agc2_enabled),
                "fixed_gain_db": float(agc2_fixed_gain),
                "adaptive_digital": bool(agc2_adaptive),
                "adaptive_headroom_db": float(agc2_headroom),
                "adaptive_max_gain_db": float(agc2_max_gain),
                "adaptive_initial_gain_db": float(agc2_initial_gain),
                "adaptive_max_gain_change_db_per_second": float(agc2_change_rate),
                "adaptive_max_output_noise_level_dbfs": float(agc2_noise_floor),
            },
            "gain_adjustment": {
                "enabled": bool(gain_adjustment_enabled),
                "pre_gain_factor": db_to_linear(pre_gain_db),
                "post_gain_factor": db_to_linear(post_gain_db),
            },
            "high_pass_filter_enabled": bool(high_pass_filter_enabled),
        },
    }
    validate(config)
    return config


def yaml_text(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, allow_unicode=True, sort_keys=False)


def flatten(config: dict[str, Any]) -> tuple[Any, ...]:
    return (
        nested(config, "audio", "playback", "device"),
        nested(config, "audio", "playback", "sample_rate_hz"),
        nested(config, "audio", "playback", "channels"),
        nested(config, "audio", "capture", "device"),
        nested(config, "audio", "capture", "sample_rate_hz"),
        nested(config, "audio", "capture", "channels"),
        nested(config, "buffers", "period_ms"),
        nested(config, "buffers", "alsa_buffer_ms"),
        nested(config, "buffers", "application_buffer_ms"),
        nested(config, "apm", "aec", "enabled"),
        nested(config, "apm", "aec", "mode"),
        nested(config, "apm", "aec", "multi_channel_render"),
        nested(config, "apm", "aec", "multi_channel_capture"),
        nested(config, "apm", "delay", "base_ms"),
        nested(config, "apm", "ns", "enabled"),
        nested(config, "apm", "ns", "level"),
        nested(config, "apm", "agc1", "enabled"),
        nested(config, "apm", "agc1", "mode"),
        nested(config, "apm", "agc1", "target_level_dbfs"),
        nested(config, "apm", "agc1", "compression_gain_db"),
        nested(config, "apm", "agc1", "limiter"),
        nested(config, "apm", "agc2", "enabled"),
        nested(config, "apm", "agc2", "fixed_gain_db"),
        nested(config, "apm", "agc2", "adaptive_digital"),
        nested(config, "apm", "agc2", "adaptive_headroom_db"),
        nested(config, "apm", "agc2", "adaptive_max_gain_db"),
        nested(config, "apm", "agc2", "adaptive_initial_gain_db"),
        nested(config, "apm", "agc2", "adaptive_max_gain_change_db_per_second"),
        nested(config, "apm", "agc2", "adaptive_max_output_noise_level_dbfs"),
        nested(config, "apm", "gain_adjustment", "enabled"),
        linear_to_db(nested(config, "apm", "gain_adjustment", "pre_gain_factor")),
        linear_to_db(nested(config, "apm", "gain_adjustment", "post_gain_factor")),
        nested(config, "apm", "high_pass_filter_enabled"),
    )


def read_config(path_text: str) -> tuple[Any, ...]:
    path = Path(path_text).expanduser().resolve()
    try:
        with path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, dict):
            raise ValueError("YAML根节点必须是映射")
        # sdk-config.yaml keeps the complete chain.  The APM page edits only
        # the common audio/buffer/apm portion and preserves model/output keys.
        if "apm" in config:
            config = {"audio": config.get("audio", DEFAULTS["audio"]),
                      "buffers": config.get("buffers", DEFAULTS["buffers"]),
                      "apm": config.get("apm", DEFAULTS["apm"])}
        # Rebuild through the same strict field mapping used for saving.
        values = flatten(config)
        normalized = make_config(*values)
        return (*flatten(normalized), yaml_text(normalized), f"已加载：{path}")
    except Exception as exc:
        raise gr.Error(f"加载失败：{exc}") from exc


def read_default_config() -> tuple[Any, ...]:
    """Load factory defaults into the APM form without changing the live file."""
    values = read_config(str(FACTORY_DEFAULT_CONFIG))
    return (*values[:-1], f"已加载默认配置：{FACTORY_DEFAULT_CONFIG}（尚未保存）")


def preview_config(*values: Any) -> str:
    try:
        return yaml_text(make_config(*values))
    except Exception as exc:
        raise gr.Error(f"配置无效：{exc}") from exc


def save_config(path_text: str, *values: Any) -> tuple[str, str]:
    path = Path(path_text).expanduser().resolve()
    try:
        if path == FACTORY_DEFAULT_CONFIG.resolve():
            raise ValueError("default-config.yaml 是只读默认配置，不能保存覆盖")
        config = make_config(*values)
        if path.exists():
            with path.open("r", encoding="utf-8") as stream:
                root = yaml.safe_load(stream) or {}
            if not isinstance(root, dict):
                raise ValueError("YAML根节点必须是映射")
        else:
            root = {}
        root.update(config)
        config = root
        path.parent.mkdir(parents=True, exist_ok=True)
        content = yaml_text(config)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except Exception:
            Path(temporary).unlink(missing_ok=True)
            raise
        return content, f"保存成功：{path}。重启 run-alsa-apm 后生效。"
    except Exception as exc:
        raise gr.Error(f"保存失败：{exc}") from exc


def build_ui(config_path: Path) -> gr.Blocks:
    initial = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else DEFAULTS
    validate(initial)

    with gr.Blocks(title="WebRTC APM 配置") as app:
        gr.Markdown(
            "# WebRTC APM / ALSA 配置\n"
            "配置保存到 YAML；C++ 后端在启动时读取。"
        )
        config_path_box = gr.Textbox(label="配置文件路径", value=str(config_path.resolve()))

        with gr.Tab("音频与缓冲"):
            with gr.Row():
                playback_device = gr.Textbox(label="放音设备", value=nested(initial, "audio", "playback", "device"))
                playback_rate = gr.Dropdown(NATIVE_RATES, label="放音参考速率 (Hz)", value=nested(initial, "audio", "playback", "sample_rate_hz"))
                playback_channels = gr.Number(label="放音声道数", value=nested(initial, "audio", "playback", "channels"), precision=0)
            with gr.Row():
                capture_device = gr.Textbox(label="录音设备", value=nested(initial, "audio", "capture", "device"))
                capture_rate = gr.Dropdown(NATIVE_RATES, label="录音速率 (Hz)", value=nested(initial, "audio", "capture", "sample_rate_hz"))
                capture_channels = gr.Number(label="录音声道数", value=nested(initial, "audio", "capture", "channels"), precision=0)
            with gr.Row():
                period_ms = gr.Number(label="ALSA period (ms)", value=nested(initial, "buffers", "period_ms"), precision=0)
                alsa_buffer_ms = gr.Number(label="ALSA buffer (ms)", value=nested(initial, "buffers", "alsa_buffer_ms"), precision=0)
                application_buffer_ms = gr.Number(label="应用层队列深度 (ms)", value=nested(initial, "buffers", "application_buffer_ms"), precision=0)

        with gr.Tab("AEC 与延迟"):
            with gr.Row():
                aec_enabled = gr.Checkbox(label="启用AEC", value=nested(initial, "apm", "aec", "enabled"))
                aec_mode = gr.Radio(["aec3", "mobile"], label="AEC模式", value=nested(initial, "apm", "aec", "mode"))
                base_delay_ms = gr.Number(label="基础延迟 (ms)", value=nested(initial, "apm", "delay", "base_ms"), precision=0)
            gr.Markdown("实际传给APM的延迟 = 基础延迟 + ALSA放音FIFO延迟 + ALSA录音FIFO延迟（最终限制为0–500 ms）。")
            with gr.Row():
                multi_render = gr.Checkbox(label="AEC多声道放音参考", value=nested(initial, "apm", "aec", "multi_channel_render"))
                multi_capture = gr.Checkbox(label="AEC多声道录音处理", value=nested(initial, "apm", "aec", "multi_channel_capture"))
                high_pass_filter_enabled = gr.Checkbox(label="启用高通滤波器", value=nested(initial, "apm", "high_pass_filter_enabled"))

        with gr.Tab("NS"):
            with gr.Row():
                ns_enabled = gr.Checkbox(label="启用NS降噪", value=nested(initial, "apm", "ns", "enabled"))
                ns_level = gr.Radio(["low", "moderate", "high", "very_high"], label="降噪等级", value=nested(initial, "apm", "ns", "level"))

        with gr.Tab("AGC1"):
            with gr.Row():
                agc1_enabled = gr.Checkbox(label="启用AGC1", value=nested(initial, "apm", "agc1", "enabled"))
                agc1_mode = gr.Radio(["adaptive_analog", "adaptive_digital", "fixed_digital"], label="AGC1模式", value=nested(initial, "apm", "agc1", "mode"))
                agc1_limiter = gr.Checkbox(label="AGC1限幅器", value=nested(initial, "apm", "agc1", "limiter"))
            with gr.Row():
                agc1_target = gr.Slider(0, 31, step=1, label="目标电平 (-dBFS)", value=nested(initial, "apm", "agc1", "target_level_dbfs"))
                agc1_gain = gr.Slider(0, 90, step=1, label="压缩增益 (dB)", value=nested(initial, "apm", "agc1", "compression_gain_db"))
            gr.Markdown("嵌入式设备推荐 `fixed_digital` 或 `adaptive_digital`；示例未联动ALSA mixer音量。")

        with gr.Tab("AGC2"):
            with gr.Row():
                agc2_enabled = gr.Checkbox(label="启用AGC2", value=nested(initial, "apm", "agc2", "enabled"))
                agc2_fixed_gain = gr.Slider(0, 49.5, step=0.5, label="固定数字增益 (dB)", value=nested(initial, "apm", "agc2", "fixed_gain_db"))
                agc2_adaptive = gr.Checkbox(label="自适应数字增益", value=nested(initial, "apm", "agc2", "adaptive_digital"))
            with gr.Row():
                agc2_headroom = gr.Number(label="Headroom (dB)", value=nested(initial, "apm", "agc2", "adaptive_headroom_db"))
                agc2_max_gain = gr.Number(label="最大增益 (dB)", value=nested(initial, "apm", "agc2", "adaptive_max_gain_db"))
                agc2_initial_gain = gr.Number(label="初始增益 (dB)", value=nested(initial, "apm", "agc2", "adaptive_initial_gain_db"))
            with gr.Row():
                agc2_change_rate = gr.Number(label="最大增益变化 (dB/s)", value=nested(initial, "apm", "agc2", "adaptive_max_gain_change_db_per_second"))
                agc2_noise_floor = gr.Number(label="最大输出噪声电平 (dBFS)", value=nested(initial, "apm", "agc2", "adaptive_max_output_noise_level_dbfs"))

        with gr.Tab("前置/后置增益"):
            gain_adjustment_enabled = gr.Checkbox(
                label="启用前置/后置增益",
                value=nested(initial, "apm", "gain_adjustment", "enabled"),
            )
            with gr.Row():
                pre_gain_db = gr.Slider(
                    -60, 60, step=0.1, label="前置增益 (dB)",
                    value=linear_to_db(nested(initial, "apm", "gain_adjustment", "pre_gain_factor")),
                )
                post_gain_db = gr.Slider(
                    -60, 60, step=0.1, label="后置增益 (dB)",
                    value=linear_to_db(nested(initial, "apm", "gain_adjustment", "post_gain_factor")),
                )
            gr.Markdown(
                "界面使用dB；保存YAML时按 `linear = 10^(dB/20)` 转为线性倍率。"
            )

        controls = [
            playback_device, playback_rate, playback_channels,
            capture_device, capture_rate, capture_channels,
            period_ms, alsa_buffer_ms, application_buffer_ms,
            aec_enabled, aec_mode, multi_render, multi_capture, base_delay_ms,
            ns_enabled, ns_level,
            agc1_enabled, agc1_mode, agc1_target, agc1_gain, agc1_limiter,
            agc2_enabled, agc2_fixed_gain, agc2_adaptive, agc2_headroom,
            agc2_max_gain, agc2_initial_gain, agc2_change_rate, agc2_noise_floor,
            gain_adjustment_enabled, pre_gain_db, post_gain_db,
            high_pass_filter_enabled,
        ]
        with gr.Row():
            load_button = gr.Button("从磁盘重新加载")
            load_default_button = gr.Button("加载默认配置")
            preview_button = gr.Button("校验并预览")
            save_button = gr.Button("保存到YAML", variant="primary")
        yaml_preview = gr.Code(label="sdk-config.yaml", language="yaml", value=yaml_text(initial))
        status = gr.Textbox(label="状态", interactive=False)

        load_button.click(read_config, inputs=[config_path_box], outputs=[*controls, yaml_preview, status])
        load_default_button.click(read_default_config, outputs=[*controls, yaml_preview, status])
        preview_button.click(preview_config, inputs=controls, outputs=[yaml_preview])
        save_button.click(save_config, inputs=[config_path_box, *controls], outputs=[yaml_preview, status])
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if not _port_is_available(args.host, args.port):
        raise SystemExit(
            f"端口 {args.port} 已被占用。请直接打开现有页面，或使用 --port 7871 "
            f"（也可设置 GRADIO_SERVER_PORT）。"
        )
    build_ui(args.config).launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
