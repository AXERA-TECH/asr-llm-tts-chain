from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
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
            self.assertEqual(len(values), 81)
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


if __name__ == "__main__":
    unittest.main()
