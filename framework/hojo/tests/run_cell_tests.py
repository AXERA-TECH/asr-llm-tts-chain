#!/usr/bin/env python3
"""Run on the development host without AX hardware: python run_cell_tests.py."""
import os
from pathlib import Path
import subprocess
import tempfile

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
  *result = strdup("test input");
  return 0;
}
''')
    library = build / 'fake-asr.so'
    subprocess.run(['g++', '-shared', '-fPIC', str(asr), '-o', str(library)], check=True)
    for scenario in ['normal', 'parts', 'mismatch', 'error', 'disconnect', 'empty',
                     'prepare_error', 'synth_error', 'disabled']:
        root = build / scenario
        root.mkdir()
        (root / 'fake-asr.so').symlink_to(library)
        env = dict(os.environ, CELL_TEST_ROOT=str(root), CELL_TEST_SCENARIO=scenario)
        subprocess.run([str(binary), str(root), str(tests / 'fake_worker.py'), scenario],
                       env=env, check=True, timeout=20)
        events = (root / 'events').read_text().splitlines()
        boundary = events.index('ASR 2')
        first = events[:boundary]
        second = events[boundary:]
        assert first[0] == 'ASR 1' and second[0] == 'ASR 2', events
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
