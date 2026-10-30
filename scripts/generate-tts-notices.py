#!/usr/bin/env python3
"""Pre-render language/voice-specific slow-down notices on the AX650 board."""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import wave

import yaml

SDK = Path(__file__).resolve().parents[1]


def valid_wav(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as wav:
            return wav.getnframes() > 0 and wav.getsampwidth() == 2
    except (OSError, EOFError, wave.Error):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SDK / "config/sdk-config.yaml")
    parser.add_argument("--voices", type=int, nargs="+", default=list(range(15)))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if any(voice < 0 or voice > 14 for voice in args.voices):
        parser.error("voices must be in [0, 14]")
    config = yaml.safe_load(args.config.read_text())
    tts = config["hojo_tts"]
    resolve = lambda path: (SDK / path).resolve()
    directory = resolve(tts.get("notice_dir", "models-for-asr-chain/tts/hojo/notices"))
    directory.mkdir(parents=True, exist_ok=True)
    needed = [
        (voice, directory / f'{"en" if voice < 13 else "zh"}-{voice}.wav')
        for voice in dict.fromkeys(args.voices)
    ]
    needed = [(voice, path) for voice, path in needed if args.force or not valid_wav(path)]
    if not needed:
        print("All requested notices already exist.")
        return
    model = resolve(tts["model_path"])
    env = dict(os.environ)
    libs = [SDK / "build", Path("/soc/lib"), Path("/opt/lib")]
    env["LD_LIBRARY_PATH"] = ":".join(map(str, libs)) + ":" + env.get("LD_LIBRARY_PATH", "")
    env.setdefault("OMP_NUM_THREADS", "1")
    command = [str(resolve(tts["runner"])), "--server", str(model), "17659",
               str(model / "fine_local.axmodel"), str(model / "decoder_sq.axmodel"),
               str(model / "speaker_vecs.bin"), str(model / "id2code.bin")]
    # The generation process uses the same lock as the production chain.
    with Path(config["llm"]["npu_lock_file"]).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   text=True, env=env, cwd=SDK)
        try:
            def receive(expected: str) -> None:
                for line in process.stdout:
                    if line.rstrip() == expected:
                        return
                    if line.startswith("ERR "):
                        raise RuntimeError(line.rstrip())
                raise RuntimeError("TTS worker exited before " + expected)

            receive("READY PREPARE_V1")
            for voice, path in needed:
                text = "Please speak more slowly, I can’t keep up." if voice < 13 else "请慢点说，我跟不上了。"
                temporary = path.with_suffix(".tmp.wav")
                # std::quoted uses backslash escapes for quotes and backslashes.
                quoted = '"' + str(temporary).replace('\\', '\\\\').replace('"', '\\"') + '"'
                process.stdin.write(f"PREPARE 1 {voice} {text}\n")
                process.stdin.flush()
                receive("OK")
                process.stdin.write(f'SYNTH 1 {tts["max_new_tokens"]} {quoted}\n')
                process.stdin.flush()
                receive("OK")
                if not valid_wav(temporary):
                    raise RuntimeError(f"Invalid notice WAV: {temporary}")
                temporary.replace(path)
                print(f"Generated {path.name}", flush=True)
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process.stdout.close()
        if process.returncode:
            raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
