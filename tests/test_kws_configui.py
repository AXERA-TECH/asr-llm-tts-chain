from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class KwsConfigUiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_ui = load_module(
            "test_sdk_config_ui", ROOT / "framework/configui/sdk-config-ui.py"
        )
        cls.runtime_ui = load_module(
            "test_runtime_ui", ROOT / "framework/configui/runtime-ui.py"
        )

    def test_kws_form_round_trip_keeps_other_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "sdk-config.yaml"
            shutil.copy(ROOT / "config/sdk-config.yaml", target)
            original = self.config_ui._read(str(target))
            values = list(self.config_ui._controls_from_config(original))
            self.assertEqual(len(values), 87)
            values[16:19] = [0.73, 5, 1200]

            app = self.config_ui.build_ui(target)
            try:
                content, _ = app.fns[0].fn(str(target), *values)
            finally:
                app.close()
            saved = yaml.safe_load(content)

            self.assertEqual(
                (saved["kws"]["threshold"], saved["kws"]["stable_frames"],
                 saved["kws"]["cooldown_ms"]),
                (0.73, 5, 1200),
            )
            for section in ("llm", "hojo_tts", "runtime"):
                self.assertEqual(saved[section], original[section])

    def test_tts_and_filter_controls_round_trip_and_validate(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "sdk-config.yaml"
            shutil.copy(ROOT / "config/sdk-config.yaml", target)
            values = list(self.config_ui._controls_from_config(self.config_ui._read(str(target))))
            values[81:87] = [3, 0.25, 30, 12, 4, "notices with spaces"]
            app = self.config_ui.build_ui(target)
            try:
                content, _ = app.fns[0].fn(str(target), *values)
                saved = yaml.safe_load(content)
                self.assertEqual(saved["llm"]["min_asr_characters"], 3)
                self.assertEqual(saved["llm"]["max_asr_punctuation_ratio"], 0.25)
                self.assertEqual(saved["hojo_tts"]["min_chinese_characters"], 30)
                self.assertEqual(saved["hojo_tts"]["min_english_words"], 12)
                self.assertEqual(saved["hojo_tts"]["backlog_threshold"], 4)
                self.assertEqual(saved["hojo_tts"]["notice_dir"], "notices with spaces")
                for index, bad in [(81, -1), (82, 0), (82, float("nan")), (83, 0), (84, 0), (85, -1), (86, "")]:
                    invalid = values.copy()
                    invalid[index] = bad
                    with self.assertRaises(Exception):
                        app.fns[0].fn(str(target), *invalid)
                self.assertEqual(target.read_text(), content)
            finally:
                app.close()

    def test_tts_status_active_stopped_stale_and_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "queue-status.json"
            status = {"pending_frames": 5, "text_frames": 3, "playback_frames": 2,
                      "threshold": 8, "overflows": 1, "running": True,
                      "updated_unix_ms": time.time() * 1000}
            path.write_text(json.dumps(status))
            self.assertIn("积压子帧：5", self.runtime_ui.read_tts_queue(root))
            self.assertIn("待播放 2", self.runtime_ui.read_tts_queue(root))
            status["running"] = False
            path.write_text(json.dumps(status))
            self.assertIn("已停止", self.runtime_ui.read_tts_queue(root))
            status.update(running=True, updated_unix_ms=0)
            path.write_text(json.dumps(status))
            self.assertIn("已过期", self.runtime_ui.read_tts_queue(root))
            path.write_text("{")
            self.assertIn("暂无", self.runtime_ui.read_tts_queue(root))

    def test_runtime_snapshot_contains_kws(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = yaml.safe_load(
                (ROOT / "config/sdk-config.yaml").read_text()
            )
            paths = {
                "kws": root / "kws.txt",
                "asr": root / "asr.txt",
                "llm": root / "llm.txt",
                "tts": root / "tts",
            }
            paths["kws"].write_text("[time] KWS: yes confidence=90.00%\n")
            paths["asr"].write_text("asr\n")
            paths["llm"].write_text("[time] Qwen3: answer\n")
            paths["tts"].mkdir()
            config["kws"]["output_file"] = str(paths["kws"])
            config["output"]["text_file"] = str(paths["asr"])
            config["llm"]["output_file"] = str(paths["llm"])
            config["hojo_tts"]["output_dir"] = str(paths["tts"])
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
            )

            outputs = self.runtime_ui.load_runtime_outputs(config_path, root)
            self.assertEqual(
                self.runtime_ui.read_snapshot(outputs)[:3],
                ("[time] KWS: yes confidence=90.00%\n",
                 "asr\n", "[time] Qwen3: answer"),
            )

    def test_old_config_migration_preserves_settings_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sdk.yaml"
            original = yaml.safe_load((ROOT / "config/sdk-config.yaml").read_text())
            original["llm"]["system_prompt"] = "你是一个乐于助人的助手。"
            for key in ("min_asr_characters", "max_asr_punctuation_ratio"):
                del original["llm"][key]
            for key in ("min_chinese_characters", "min_english_words", "backlog_threshold", "notice_dir"):
                del original["hojo_tts"][key]
            path.write_text(yaml.safe_dump(original, allow_unicode=True, sort_keys=False))
            command = ["bash", str(ROOT / "scripts/migrate-config.sh"), str(path)]
            subprocess.run(command, check=True, capture_output=True)
            migrated = yaml.safe_load(path.read_text())
            self.assertEqual(migrated["audio"], original["audio"])
            self.assertEqual(migrated["hojo_tts"]["english_voice"], original["hojo_tts"]["english_voice"])
            self.assertEqual(migrated["llm"]["min_asr_characters"], 5)
            self.assertIn("简洁干练", migrated["llm"]["system_prompt"])
            first = path.read_bytes()
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual(path.read_bytes(), first)
            migrated["llm"]["system_prompt"] = "用户自定义提示词"
            path.write_text(yaml.safe_dump(migrated, allow_unicode=True, sort_keys=False))
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual(yaml.safe_load(path.read_text())["llm"]["system_prompt"], "用户自定义提示词")


if __name__ == "__main__":
    unittest.main()
