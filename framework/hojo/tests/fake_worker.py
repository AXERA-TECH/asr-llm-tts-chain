#!/usr/bin/env python3
"""Model-free protocol fixture. Logs observable scheduling and overlap."""
import fcntl
import os
from pathlib import Path
import shlex
import struct
import sys
import time
import wave

root = Path(os.environ['CELL_TEST_ROOT'])
scenario = os.environ['CELL_TEST_SCENARIO']


def event(text):
    with (root / 'events').open('a') as f:
        f.write(text + '\n')


def assert_cell_locked():
    with (root / 'npu.lock').open('r') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        raise AssertionError('Cell released its cross-process NPU lock early')


def read_exact(n):
    data = b''
    while len(data) < n:
        part = sys.stdin.buffer.read(n - len(data))
        if not part:
            return None
        data += part
    return data


if '--server' in sys.argv:
    cache = {}
    print('READY PREPARE_V1', flush=True)
    for line in sys.stdin:
        op = line.split()[0]
        if op == 'PREPARE':
            _, ident, voice, text = line.rstrip('\n').split(' ', 3)
            event(f'PREPARE {text}')
            if scenario == 'prepare_error' and not (root / 'failed').exists():
                (root / 'failed').touch()
                print('ERR mock preparation failure', flush=True)
                continue
            cache[ident] = (voice, text)
        elif op == 'SYNTH':
            _, ident, tokens, output = shlex.split(line)
            assert_cell_locked()
            assert not (root / 'qwen-active').exists(), 'NPU stages overlapped'
            voice, text = cache.pop(ident)
            event(f'SYNTH {voice} {text}')
            if scenario == 'synth_error' and not (root / 'failed').exists():
                (root / 'failed').touch()
                print('ERR mock synthesis failure', flush=True)
                continue
            with wave.open(output, 'wb') as wav:
                wav.setparams((1, 2, 24000, 0, 'NONE', 'NONE'))
                wav.writeframes(struct.pack('<240h', *([100] * 240)))
        elif op == 'CLEAR':
            cache.clear()
        else:
            raise AssertionError(line)
        print('OK', flush=True)
else:
    fd = int(sys.argv[sys.argv.index('--response-fd') + 1])

    def frame(kind, text):
        payload = text.encode()
        os.write(fd, kind.encode() + struct.pack('=I', len(payload)) + payload)

    frame('R', 'ready')
    count = 0
    while True:
        size = read_exact(4)
        if size is None:
            break
        assert read_exact(struct.unpack('=I', size)[0]) is not None
        count += 1
        assert_cell_locked()
        event(f'QBEGIN {count}')
        (root / 'qwen-active').touch()
        text = '[zh]你好，世界' if count == 1 else '[en]Hello,world'
        if scenario in ('error', 'disconnect', 'empty', 'prepare_error', 'synth_error') and count == 1:
            text = 'partial,'
        if scenario == 'mismatch' and count == 1:
            text = '[zh]错误，'
        if scenario != 'disabled':
            assert '--stream-tokens' in sys.argv
            previous = (root / 'events').read_text().count('PREPARE ')
            frame('T', text)
            deadline = time.monotonic() + 5
            while (root / 'events').read_text().count('PREPARE ') == previous:
                assert time.monotonic() < deadline, 'tokenizer did not overlap Qwen'
                time.sleep(.005)
        (root / 'qwen-active').unlink()
        event(f'QDONE {count}')
        if scenario == 'disconnect' and count == 1:
            sys.exit(0)  # EOF before D: partial preparation must be discarded.
        if scenario == 'error' and count == 1:
            frame('E', 'mock failure')
        else:
            if scenario == 'mismatch' and count == 1:
                text = '[en]Correct,tail'
            if scenario == 'empty' and count == 1:
                text = ''
            frame('D', text)
