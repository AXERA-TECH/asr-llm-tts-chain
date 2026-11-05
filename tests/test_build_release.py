"""Exercise release publication with fake compilers and real Git/tar in isolation."""

import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = {
    "audio-pipeline", "run-alsa-apm", "libfastenhance.so",
    "qwen3-worker", "hojo-tts-resident",
}
FAKE_TOOL = r'''#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
import sys

args = sys.argv[1:]
tool = Path(sys.argv[0]).name
if tool == "tar":
    if os.environ.get("FAIL_TAR") == args[0]:
        if args[0] == "-czf":
            Path(args[1]).write_text("incomplete archive")
        sys.exit(7)
    sys.exit(subprocess.call([os.environ["REAL_TAR"], *args]))
if tool == "fake-cxx":
    if os.environ.get("FAIL_COMPILE"):
        sys.exit(8)
    target = Path(args[args.index("-o") + 1])
elif args[0] == "--build":
    name = args[args.index("--target") + 1]
    if name == "kaldi-native-fbank-core" or name == os.environ.get("MISSING_TARGET"):
        sys.exit(0)
    target = Path(args[1]).parent / name
else:
    sys.exit(0)
target.write_text("fake compiled artifact\n")
target.chmod(0o755)
'''


class BuildReleaseTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.repo = base / "repo with spaces"
        self.repo.mkdir()
        for relative in ("scripts/build.sh", "driver/ns/build-fastenhance.sh", ".gitignore"):
            dest = self.repo / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, dest)
        self.git("init", "-q")
        self.git("add", ".")
        self.git("-c", "user.name=Build Test", "-c", "user.email=build@example.invalid",
                 "commit", "-qm", "fixture")
        self.commit = self.git("rev-parse", "HEAD").strip()
        self.build = self.repo / "build"
        cache = self.build / "campplus-third_party/kaldi-native-fbank-1.22.3/CMakeLists.txt"
        cache.parent.mkdir(parents=True)
        cache.write_text("cached third-party source\n")
        tools = base / "tools"
        tools.mkdir()
        for name in ("fake-cxx", "cmake", "tar"):
            tool = tools / name
            tool.write_text(FAKE_TOOL)
            tool.chmod(0o755)
        sdk = base / "sdk"
        (sdk / "include/alsa").mkdir(parents=True)
        (sdk / "include/alsa/asoundlib.h").touch()
        (sdk / "lib").mkdir()
        (sdk / "lib/libasound.so").touch()
        self.env = os.environ.copy()
        self.env.update(
            PATH=f"{tools}:{os.environ['PATH']}",
            CC=str(tools / "fake-cxx"), CXX=str(tools / "fake-cxx"),
            AX650_SDK_ROOT=str(sdk), CROSS_SYSROOT="", ALSA_SYSROOT="",
            ALSA_INCLUDE_DIR=str(sdk / "include"), ALSA_LIBRARY=str(sdk / "lib/libasound.so"),
            REAL_TAR=shutil.which("tar"),
        )
        self.old = self.repo / "build_release_20260101-000000_old.tar.gz"
        self.legacy = self.repo / "build_release.tar.gz"
        self.unrelated = self.repo / "backup.tar.gz"
        for archive in (self.old, self.legacy, self.unrelated):
            archive.write_bytes(b"previous archive")

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], text=True)

    def run_build(self, **env):
        return subprocess.run(
            ["bash", str(self.repo / "scripts/build.sh")], cwd=self.repo.parent,
            env={**self.env, **env}, capture_output=True, text=True,
        )

    def assert_old_preserved(self):
        for archive in (self.old, self.legacy, self.unrelated):
            self.assertEqual(archive.read_bytes(), b"previous archive")
        self.assertEqual(list(self.repo.glob("build_release_*.tar.gz")), [self.old])
        self.assertEqual(list(self.repo.glob(".build-release.*")), [])

    def test_complete_build_publishes_versioned_package_and_cleans_history(self):
        result = self.run_build()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        archives = list(self.repo.glob("build_release_*.tar.gz"))
        self.assertEqual(len(archives), 1)
        self.assertRegex(archives[0].name,
                         rf"^build_release_\d{{8}}-\d{{6}}_{self.commit[:12]}\.tar\.gz$")
        with tarfile.open(archives[0]) as archive:
            self.assertEqual(set(archive.getnames()),
                             {f"build/{name}" for name in ARTIFACTS | {"VERSION"}})
            version = archive.extractfile("build/VERSION").read().decode()
            self.assertIn(f"commit={self.commit}\n", version)
            self.assertRegex(version, r"build_date=\d{4}-\d{2}-\d{2}T.*[+-]\d{4}\n")
            for name in ARTIFACTS:
                self.assertEqual(archive.extractfile(f"build/{name}").read(),
                                 b"fake compiled artifact\n")
        self.assertFalse(self.old.exists())
        self.assertFalse(self.legacy.exists())
        self.assertTrue(self.unrelated.exists())
        self.assertEqual(list(self.repo.glob(".build-release.*")), [])

    def test_tracked_edits_mark_package_dirty(self):
        with (self.repo / "scripts/build.sh").open("a") as script:
            script.write("\n# local modification\n")
        result = self.run_build()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(next(self.repo.glob("build_release_*.tar.gz")).name.endswith("-dirty.tar.gz"))
        self.assertIn(f"git_version={self.commit[:12]}-dirty\n",
                      (self.build / "VERSION").read_text())

    def test_compression_or_validation_failure_preserves_history(self):
        for mode in ("-czf", "-tzf"):
            with self.subTest(mode=mode):
                result = self.run_build(FAIL_TAR=mode)
                self.assertNotEqual(result.returncode, 0)
                self.assert_old_preserved()

    def test_skipped_components_do_not_package_stale_outputs(self):
        for name in ARTIFACTS:
            (self.build / name).write_text("stale artifact\n")
            (self.build / name).chmod(0o755)
        for overrides in ({"AX650_SDK_ROOT": "/nonexistent-sdk"},
                          {"ALSA_INCLUDE_DIR": "/nonexistent-alsa"}):
            with self.subTest(overrides=overrides):
                result = self.run_build(**overrides)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("release packaging skipped", result.stderr)
                self.assert_old_preserved()
        self.assertFalse((self.build / "VERSION").exists())

    def test_missing_output_prevents_publication(self):
        result = self.run_build(MISSING_TARGET="hojo-tts-resident")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing or empty build/hojo-tts-resident", result.stderr)
        self.assert_old_preserved()

    def test_compile_failure_preserves_history(self):
        result = self.run_build(FAIL_COMPILE="1")
        self.assertNotEqual(result.returncode, 0)
        self.assert_old_preserved()


if __name__ == "__main__":
    unittest.main()
