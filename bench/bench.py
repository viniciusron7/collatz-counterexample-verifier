#!/usr/bin/env python3
"""Benchmark of the Collatz verifier (run by `make bench` in the project directory).

Run a fixed set of inputs with `-r` and show the best `time` field (the computation time
measured by the program itself, without parsing or output) over a few runs, in the
default (down to 1) and `-s` (up to the stopping time) modes.

Inputs: structured 2^N − 1 and random (fixed seed) of N bits, via standard input.
The `time` field has a resolution of 1 µs, so small inputs are repeated and the
average per number is reported. In the 64- and 128-bit batches (distinct numbers) each
computation takes less than 1 µs and the average of the rounded fields is only indicative;
there the comparison (`speedup`) uses the wall time per number, which includes parsing and output.

Usage: python3 bench/bench.py [--bin ./collatz] [--ref OTHER_BINARY] [--reps 3]
                              [--max-bits 10000000] [--quick]
"""
import argparse
import os
import random
import re
import subprocess
import sys
import tempfile
import time

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)

# The programs record every verified number in Tested/; the benchmark uses a temporary directory
# so that the real file is never touched.
import atexit as _atexit, shutil as _shutil  # noqa: E402
_TESTED_TMP = tempfile.mkdtemp(prefix="collatz_tested_")
os.environ["COLLATZ_TESTED"] = _TESTED_TMP
_atexit.register(_shutil.rmtree, _TESTED_TMP, True)

TIME = re.compile(r' time=([0-9.]+)')


def run(binary, args, stdin_path=None):
    """Run and return (output without the time fields, list of times, wall seconds)."""
    inp = open(stdin_path) if stdin_path else subprocess.DEVNULL
    t0 = time.perf_counter()
    p = subprocess.run([binary, '-r'] + args, stdin=inp, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if stdin_path:
        inp.close()
    if p.returncode not in (0, 2):
        sys.exit(f'error: {binary} {" ".join(args)} exited with status {p.returncode}\n{p.stderr}')
    return TIME.sub('', p.stdout), [float(t) for t in TIME.findall(p.stdout)], wall


def fmt(sec):
    if sec < 1e-3:
        return f'{sec * 1e6:8.2f} µs'
    if sec < 1:
        return f'{sec * 1e3:8.2f} ms'
    return f'{sec:8.3f} s '


def make_inputs(tmp, max_bits, quick):
    """List of cases: (name, input file, number of numbers in the file)."""
    rnd = random.Random(20261020)
    cases = []

    def add(name, lines):
        path = os.path.join(tmp, f'case{len(cases)}.txt')
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        cases.append((name, path, len(lines)))

    for bits, count in ((64, 50000 if quick else 300000), (128, 20000 if quick else 100000)):
        add(f'batch {bits} bits', [str(rnd.getrandbits(bits) | (1 << (bits - 1))) for _ in range(count)])
    for bits in [b for b in (1000, 10000, 100000, 1000000, 10000000) if b <= max_bits]:
        copies = 200 if bits <= 1000 else 20 if bits <= 10000 else 1   # copies: 1 µs resolution
        add(f'2^{bits}-1', [f'2^{bits}-1'] * copies)
        add(f'random {bits} bits', [hex(rnd.getrandbits(bits) | (1 << (bits - 1)) | 1)] * copies)
    return cases


def measure(binary, mode, case, reps):
    name, path, count = case
    best_core = best_wall = float('inf')
    out = None
    for _ in range(reps):
        o, ts, wall = run(binary, mode, path)
        if out is not None and o != out:
            sys.exit(f'error: output changed between runs in {name}')
        out = o
        best_core = min(best_core, sum(ts) / count)
        best_wall = min(best_wall, wall / count)
    return out, best_core, best_wall


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin', default='./collatz', help='binary to measure (default: ./collatz)')
    ap.add_argument('--ref', help='another binary to compare side by side (e.g. the previous version)')
    ap.add_argument('--reps', type=int, default=3, help='runs per case; the best counts (default 3)')
    ap.add_argument('--max-bits', type=int, default=10000000, help='largest size measured (default 10^7)')
    ap.add_argument('--quick', action='store_true', help='smaller batches and up to 10^6 bits')
    a = ap.parse_args()
    bins = [os.path.abspath(a.bin)] + ([os.path.abspath(a.ref)] if a.ref else [])
    for b in bins:
        if not os.access(b, os.X_OK):
            sys.exit(f'error: {b} does not exist or is not executable (run `make` first)')
    max_bits = min(a.max_bits, 1000000) if a.quick else a.max_bits

    with tempfile.TemporaryDirectory() as tmp:
        cases = make_inputs(tmp, max_bits, a.quick)
        head = f'{"input":<24}{"mode":<6}' + ''.join(f'{os.path.basename(b):>14}' for b in bins)
        if a.ref:
            head += f'{"speedup":>9}'
        print(head)
        print('-' * len(head))
        t_start = time.perf_counter()
        for mode in ([], ['-s']):
            for case in cases:
                res = [measure(b, mode, case, a.reps) for b in bins]
                core = [r[1] for r in res]
                line = f'{case[0]:<24}{"-s" if mode else "-":<6}' + ''.join(f'{fmt(c):>14}' for c in core)
                if a.ref:
                    if case[2] >= 1000:   # batch: compare by wall time per number
                        line += f'{res[1][2] / res[0][2]:8.2f}x'
                    elif min(core) >= 2e-6 or (case[2] >= 20 and min(core) >= 5e-7):
                        line += f'{core[1] / core[0]:8.2f}x'
                    else:                 # below the resolution of the time field
                        line += f'{"—":>9}'
                    if res[0][0] != res[1][0]:
                        line += '   <<< OUTPUTS DIFFER'
                if case[2] >= 1000:
                    line += '   (average per number; wall: ' + ', '.join(fmt(r[2]).strip() for r in res) + ')'
                print(line, flush=True)
        print(f'\n(best of {a.reps} runs; {time.perf_counter() - t_start:.1f} s in total)')


if __name__ == '__main__':
    main()
