#!/usr/bin/env python3
"""Run on the development host without AX hardware: python run_cell_tests.py."""
import os
from pathlib import Path
import subprocess
import tempfile
import struct
import wave

sdk = Path(__file__).resolve().parents[3]
tests = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='asr-cell-tests-') as directory:
    build = Path(directory)
    binary = build / 'cell-test'
    includes = ['framework', 'driver/ns/include', 'driver/vad/include',
                'driver/asr/sensevoice/include', 'driver/campplus/include']
    subprocess.run(['g++', '-std=c++17', '-O1', '-g', '-ffunction-sections',
                    '-fdata-sections', *(f'-I{sdk / p}' for p in includes),
                    str(tests / 'cell_test.cpp'), '-Wl,--gc-sections',
                    '-pthread', '-ldl', '-o', str(binary)], check=True)
    asr = build / 'asr.cpp'
    asr.write_text('''
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
extern "C" void* AX_ASR_Init(int, const char*) { return (void*)1; }
extern "C" void AX_ASR_Uninit(void*) {}
extern "C" int AX_ASR_RunFile(void*, const char*, const char*, char** result) {
  static int count = 0;
  std::ofstream(std::string(std::getenv("CELL_TEST_ROOT")) + "/events", std::ios::app)
      << "ASR " << ++count << "\\n";
  const std::string scenario = std::getenv("CELL_TEST_SCENARIO");
  *result = strdup(scenario == "filtered" && count == 1 ?
      "[ASR_ERROR] sensevoice unavailable\\n[time] speaker0@0-1 ，。！？…——“”　" :
      scenario == "translation_en" ? "你好世界朋友" : "test input");
  return 0;
}
''')
    library = build / 'fake-asr.so'
    subprocess.run(['g++', '-shared', '-fPIC', str(asr), '-o', str(library)], check=True)
    for scenario in ['normal', 'parts', 'mismatch', 'error', 'disconnect', 'empty',
                     'prepare_error', 'synth_error', 'disabled', 'filtered',
                     'translation_zh', 'translation_en', 'overflow_zh',
                     'overflow_en', 'overflow_audio', 'overflow_fallback']:
        root = build / scenario
        root.mkdir()
        (root / 'fake-asr.so').symlink_to(library)
        (root / 'notices').mkdir()
        for voice in (9, 13):
            if scenario == 'overflow_fallback':
                continue
            with wave.open(str(root / 'notices' / f'{"zh" if voice == 13 else "en"}-{voice}.wav'), 'wb') as wav:
                wav.setparams((1, 2, 24000, 0, 'NONE', 'NONE'))
                wav.writeframes(struct.pack('<240h', *([222] * 240)))
        env = dict(os.environ, CELL_TEST_ROOT=str(root), CELL_TEST_SCENARIO=scenario)
        subprocess.run([str(binary), str(root), str(tests / 'fake_worker.py'), scenario],
                       env=env, check=True, timeout=20)
        events = (root / 'events').read_text().splitlines()
        boundary = events.index('ASR 2')
        first = events[:boundary]
        second = events[boundary:]
        assert first[0] == 'ASR 1' and second[0] == 'ASR 2', events
        if scenario == 'filtered':
            assert first == ['ASR 1'], events
            assert second.count('PLAY') == 2, events
            continue
        if scenario.startswith('overflow'):
            notice_phase = second if scenario == 'overflow_audio' else first
            assert notice_phase.count('NOTICE') == 1, events
            assert sum(e.startswith('SYNTH') for e in notice_phase) == (1 if scenario == 'overflow_fallback' else 0), events
            assert not any(e == 'PLAY' for e in notice_phase), events
            assert (root / 'tts with spaces' / 'queue-status.json').is_file()
            continue
        if scenario != 'disabled':
            assert next(i for i, e in enumerate(first) if e.startswith('PREPARE')) < first.index('QDONE 1')
            expected = 0 if scenario in ('error', 'disconnect', 'empty', 'prepare_error', 'synth_error') else 2
            assert first.count('PLAY') == expected, events
            assert second.count('PLAY') == (0 if scenario == 'disconnect' else 2), events
            # Every chunk is submitted to playback before the next SYNTH.
            for phase in [first, second]:
                indices = [i for i, e in enumerate(phase) if e.startswith('SYNTH')]
                for a, b in zip(indices, indices[1:]):
                    assert 'PLAY' in phase[a:b], events
        if scenario == 'mismatch':
            spoken = [e for e in first if e.startswith('SYNTH')]
            assert spoken == ['SYNTH 9 Correct,', 'SYNTH 9 tail'], events
    print('All Cell scheduling, tokenizer overlap and playback tests passed.')
