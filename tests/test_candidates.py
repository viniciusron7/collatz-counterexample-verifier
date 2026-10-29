#!/usr/bin/env python3
"""
Tests of the `candidates` generator.

    python3 tests/test_candidates.py            default suite (~1 min)
    python3 tests/test_candidates.py --long     larger windows and larger K
    python3 tests/test_candidates.py --only certificates,tables

What is checked:
  certificates  every discarded n comes with an m; a step-by-step simulation (Python) confirms
                1 ≤ m < n and T^i(n) = T^j(m). This is what guarantees that the smallest
                counterexample is never discarded.
  tables        the survivors (mod 2^K, mod 3^L and of the joint table 2^16 × 3^J) and the
                per-level counts are exactly those of tests/ref_candidates.py (full search,
                without the shortcuts of the C code).
  known         N mod 32 ∈ {7,15,27,31}, N mod 9 ∈ {0,1,3,6,7}, the 20 classes mod 288,
                N ≡ 15 (mod 64) discarded, descent-only sieve = OEIS A076227.
  generation    the generated list is exactly the set of candidates of --witnesses, in
                increasing order, in windows that cross 2^K blocks; -q, --count.
  small         intervals below 2^71 (the sieve is valid from A on): valid certificates and
                no candidate of the reference is discarded.
  consistency   --no-prune and -t 1 give identical results; --explain checks the certificate.
  cli           input errors.
  tested        collatz stores in Tested/ only the NO verdicts in [2^71, 2^127), in bulk, with
                several processes at once and when interrupted; tested.bin stays sorted
                (checked by an independent reader, tests/tested_fmt.py); candidates does not
                generate these numbers again; truncated records, wrong CRCs and corrupt blocks
                never exclude anything; migration of tested.txt; --data checked by step-by-step
                simulation; Ctrl+C, kill and "| head" stop at a safe point.
  random        -a N: N distinct numbers, sorted, in the interval, all candidates, none tested;
                --seed reproduces; uniformity by chi-square (rejection and selection).
"""

import argparse
import os
import random
import re
import sqlite3
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
# The programs record every verified number in Tested/; the tests use a temporary directory so
# that the real file is never touched.
import atexit as _atexit, shutil as _shutil  # noqa: E402
_TESTED_TMP = tempfile.mkdtemp(prefix="collatz_tested_")
os.environ["COLLATZ_TESTED"] = _TESTED_TMP
_atexit.register(_shutil.rmtree, _TESTED_TMP, True)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ref_candidates as ref  # noqa: E402

BIN = os.path.join(ROOT, "candidates")
V71 = 2 ** 71
FAILS = []
COUNTS = {}


def run(args, check=True, timeout=600):
    p = subprocess.run([BIN] + [str(a) for a in args], capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(f"candidates {' '.join(map(str, args))} -> exit code {p.returncode}\n{p.stderr}")
    return p


def ok(cat, cond, detail):
    COUNTS.setdefault(cat, [0, 0])
    if cond:
        COUNTS[cat][0] += 1
    else:
        COUNTS[cat][1] += 1
        if len(FAILS) < 30:
            FAILS.append(f"[{cat}] {detail}")


def witnesses(A, B, K=None, L=None, J=None):
    args = ["--witnesses", A, B]
    if K is not None:
        args += ["-k", K]
    if L is not None:
        args += ["-l", L]
    if J is not None:
        args += ["-j", J]
    p = run(args, check=False)
    rows = []
    for line in p.stdout.splitlines():
        f = line.split()
        if f[1] == "C":
            rows.append((int(f[0]), None))
        else:
            rows.append((int(f[0]), (int(f[2]), int(f[3]), int(f[4]))))
    return p.returncode, rows


def generated(A, B, K=None, L=None, extra=(), J=None):
    args = [A, B] + list(extra)
    if K is not None:
        args += ["-k", K]
    if L is not None:
        args += ["-l", L]
    if J is not None:
        args += ["-j", J]
    return [int(x) for x in run(args).stdout.split()]


def level_counts(K, L, extra=(), J=0):
    """Live classes per level in the table printed by -e."""
    out = run(["-e", "-k", K, "-l", L, "-j", J] + list(extra)).stdout
    alive = {}
    for line in out.splitlines():
        m = re.match(r"\s*(\d+)\s+([\d,]+)\s+[\d.]+%", line)
        if m:
            alive[int(m.group(1))] = int(m.group(2).replace(",", ""))
    m3 = re.search(r"^3\^\d+ sieve .*?: ([\d,]+) of", out, re.M)
    return alive, int(m3.group(1).replace(",", ""))


def is_no(line):
    """Is this collatz output line the verdict NO (not NOTMIN)?"""
    return line.split(" ", 1)[0] == "NO"


# --------------------------------------------------------------------------------------

def t_certificates(long):
    rnd = random.Random(20261021)
    cases = [(V71, V71 + 3000, 32, 14), (V71 + 2 ** 40, V71 + 2 ** 40 + 3000, 32, 14),
             (2 ** 100, 2 ** 100 + 2000, 32, 14), (2 ** 126 + 7, 2 ** 126 + 1500, 34, 19)]
    for K, L in [(12, 6), (20, 9), (28, 12)]:
        base = V71 + rnd.randrange(2 ** 60)
        cases.append((base, base + 4000, K, L))
    # joint table with J > L: joint discards are frequent
    for K, J in [(16, 9), (24, 12), (32, 6)]:
        base = V71 + rnd.randrange(2 ** 70)
        cases.append((base, base + 6000, K, 0, J))
    if long:
        for _ in range(6):
            base = V71 + rnd.randrange(2 ** 80)
            cases.append((base, base + 40000, rnd.choice([16, 24, 32, 34]), rnd.choice([0, 8, 14, 19])))
    # discards made only by the joint table (rare in a small window): check several with -x
    A = V71 + 77777 + rnd.randrange(2 ** 50)
    g0 = generated(A, A + 400000, J=0)
    gj = generated(A, A + 400000)
    killed = sorted(set(g0) - set(gj))
    ok("certificates", set(gj) <= set(g0) and len(killed) >= 30, f"joint table discarded {len(killed)} (expected ≥ 30)")
    for n in killed[:: max(1, len(killed) // 40)]:
        out = run(["-x", n]).stdout
        m = re.search(r"m = (\d+)\n\s+T\^(\d+)\(n\) = T\^(\d+)\(m\)", out)
        good = m and "joint table" in out and "Checked" in out \
            and ref.check_witness(n, int(m.group(2)), int(m.group(3)), int(m.group(1)))
        ok("certificates", bool(good), f"-x {n}: joint certificate does not check out")
        COUNTS.setdefault("_discards checked", [0, 0])[0] += 1
    for case in cases:
        A, B, K, L = case[:4]
        J = case[4] if len(case) > 4 else None
        rc, rows = witnesses(A, B, K, L, J)
        ok("certificates", rc == 0 and len(rows) == B - A, f"--witnesses {A} {B} -k {K} -l {L}: exit code {rc}, {len(rows)} lines")
        bad = [(n, w) for n, w in rows if w and not ref.check_witness(n, *w)]
        ok("certificates", not bad, f"false certificate (K={K}, L={L}): {bad[:3]}")
        nel = sum(1 for _, w in rows if w)
        COUNTS.setdefault("_discards checked", [0, 0])[0] += nel


def t_tables(long):
    configs = [(8, 4), (12, 6), (14, 9), (16, 8)]
    if long:
        configs += [(18, 10), (20, 12)]
    # joint table (k0 = min(K, 16)) against the full-search reference
    jconfigs = [(8, 2, 6), (10, 0, 5), (12, 4, 4), (18, 3, 4)]
    if long:
        jconfigs += [(12, 0, 7), (16, 6, 6)]
    for K, L, J in jconfigs:
        W = 2 ** K * 3 + 1234
        A = V71 + 999
        want = ref.candidates(A, A + W, K, L, J, V71)
        _, rows = witnesses(A, A + W, K, L, J)
        got = [n for n, w in rows if w is None]
        ok("tables", got == want, f"joint K={K} L={L} J={J}: {len(got)} candidates in C, {len(want)} in the reference")
        gen = generated(A, A + W, K, L, J=J)
        ok("tables", gen == want, f"joint K={K} L={L} J={J}: generator gave {len(gen)}, reference {len(want)}")
    for K, L in configs:
        s2, alive_ref = ref.sieve2(K, V71)
        s3 = ref.sieve3(L, V71)
        alive_c, n3 = level_counts(K, L)
        ok("tables", all(alive_c.get(k) == alive_ref[k] for k in range(1, K + 1)),
           f"K={K}: live per level C={[alive_c.get(k) for k in range(1, K + 1)]} ref={alive_ref[1:]}")
        ok("tables", n3 == len(s3), f"L={L}: 3^L live C={n3} ref={len(s3)}")
        # n-by-n classification in a window that covers every residue mod 2^K
        W = max(2 ** K, 3 ** L) + 777
        A = V71 + 12345
        _, rows = witnesses(A, A + W, K, L, 0)
        want = [n for n in range(A, A + W) if n % 2 ** K in s2 and n % 3 ** L in s3]
        got = [n for n, w in rows if w is None]
        ok("tables", got == want, f"K={K} L={L}: {len(got)} candidates in C, {len(want)} in the reference")
        gen = generated(A, A + W, K, L, J=0)
        ok("tables", gen == want, f"K={K} L={L}: generator gave {len(gen)}, reference {len(want)}")


def t_known(long):
    g = generated(V71, V71 + 32 * 50, 5, 0, J=0)
    ok("known", sorted({n % 32 for n in g}) == [7, 15, 27, 31], f"mod 32: {sorted({n % 32 for n in g})}")
    g = generated(V71, V71 + 9 * 200, 1, 2, J=0)
    ok("known", sorted({n % 9 for n in g}) == [0, 1, 3, 6, 7], f"mod 9: {sorted({n % 9 for n in g})}")
    g = generated(V71, V71 + 288 * 20, 5, 2, J=0)
    want = [7, 15, 27, 39, 63, 79, 91, 111, 123, 127, 135, 159, 187, 199, 207, 219, 223, 231, 255, 271]
    ok("known", sorted({n % 288 for n in g}) == want, f"mod 288: {sorted({n % 288 for n in g})}")
    g = generated(V71, V71 + 64 * 100, 6, 0, J=0)
    ok("known", all(n % 64 != 15 for n in g), "N ≡ 15 (mod 64) should be discarded")
    # OEIS A076227: classes mod 2^k without guaranteed descent within k steps
    a076227 = [1, 1, 2, 3, 4, 8, 13, 19, 38, 64, 128, 226, 367, 734, 1295, 2114, 4228, 7495, 14990, 27328]
    alive, _ = level_counts(20, 0, ["--descent-only"])
    ok("known", [alive[k] for k in range(1, 21)] == a076227, f"--descent-only: {[alive[k] for k in range(1, 21)]}")
    ok("known", ref.descent_only_counts(20)[1:] == a076227, "reference: A076227")
    # densities of the 3^L sieve (computed separately)
    for L, want3 in [(2, 5), (4, 44), (8, 3522)]:
        _, n3 = level_counts(3, L)
        ok("known", n3 == want3, f"3^{L}: {n3} live, expected {want3}")


def t_generation(long):
    rnd = random.Random(7)
    for K, L in [(10, 4), (12, 9), (16, 12), (32, 14)]:
        J = None if K == 32 else rnd.choice([0, 5, 9])
        for _ in range(3 if not long else 8):
            A = V71 + rnd.randrange(2 ** 50)
            if rnd.random() < 0.5:                       # window crossing a 2^K block
                A = ((A >> K) + 1 << K) - rnd.randrange(1, 3000)
            B = A + rnd.randrange(1, 6000)
            _, rows = witnesses(A, B, K, L, J)
            want = [n for n, w in rows if w is None]
            got = generated(A, B, K, L, J=J)
            ok("generation", got == want, f"[{A}, {B}) K={K} L={L} J={J}: generator {len(got)}, witnesses {len(want)}")
            ok("generation", got == sorted(got) and all(A <= n < B for n in got), "out of order or out of the interval")
            c = int(run(["-c", A, B, "-k", K, "-l", L] + (["-j", J] if J is not None else [])).stdout)
            ok("generation", c == len(want), f"--count {c} != {len(want)}")
    # -q and the default A = 2^71
    g = [int(x) for x in run(["-q", 50]).stdout.split()]
    ok("generation", len(g) == 50 and g[0] >= V71 and g == sorted(g), "-q 50 from 2^71")
    g2 = generated(V71, g[-1] + 1)
    ok("generation", g2 == g, "-q 50 differs from the interval [2^71, last]")
    ok("generation", run([V71 + 5, V71 + 5]).stdout == "", "an empty interval should generate nothing")
    big = generated(2 ** 127 - 5000, 2 ** 127)
    _, rows = witnesses(2 ** 127 - 5000, 2 ** 127)
    ok("generation", big == [n for n, w in rows if w is None], "near 2^127")
    # large multithreaded count = sum of the blocks
    A = V71 + 2 ** 33 + 99
    c_all = int(run(["-c", A, A + 2 ** 35]).stdout)
    c_parts = sum(int(run(["-c", A + i * 2 ** 33, A + (i + 1) * 2 ** 33]).stdout) for i in range(4))
    ok("generation", c_all == c_parts, f"--count in parts: {c_all} != {c_parts}")


def t_small(long):
    for A, B, K, L in [(2, 3000, 10, 4), (2, 5000, 16, 8), (1000, 9000, 12, 6), (2 ** 30, 2 ** 30 + 5000, 20, 10)]:
        rc, rows = witnesses(A, B, K, L, 0)
        _, rows_j = witnesses(A, B, K, L)            # with the default joint table: certificates only
        bad_j = [(n, w) for n, w in rows_j if w and not ref.check_witness(n, *w)]
        ok("small", not bad_j, f"[{A},{B}) K={K} joint: false certificates {bad_j[:3]}")
        bad = [(n, w) for n, w in rows if w and not ref.check_witness(n, *w)]
        ok("small", rc == 0 and not bad, f"[{A},{B}) K={K}: false certificates {bad[:3]}")
        # the reference (full search) discards at least what the C code discards
        s2, _ = ref.sieve2(K, A)
        s3 = ref.sieve3(L, A)
        ref_cand = {n for n in range(A, B) if n % 2 ** K in s2 and n % 3 ** L in s3}
        c_cand = {n for n, w in rows if w is None}
        ok("small", ref_cand <= c_cand, f"[{A},{B}): the C code discarded candidates of the reference: {sorted(ref_cand - c_cand)[:5]}")
        got = generated(A, B, K, L, J=0)
        ok("small", got == sorted(c_cand), f"[{A},{B}): generator differs from --witnesses")
        got_j = generated(A, B, K, L)
        ok("small", got_j == [n for n, w in rows_j if w is None], f"[{A},{B}): generator (joint) differs from --witnesses")
    # 1 is never listed; 27 (long trajectory) is a candidate
    g = generated(1, 100, 8, 2, J=0)
    ok("small", 1 not in g and 27 in g, f"[1,100): {g}")


def t_consistency(long):
    K = 26 if long else 22
    a = run(["-e", "-k", K]).stdout
    b = run(["-e", "-k", K, "--no-prune"]).stdout
    strip = lambda s: re.sub(r"Construction:.*", "", s)  # noqa: E731
    ok("consistency", strip(a) == strip(b), "--no-prune changed the result")
    A = V71 + 2 ** 45 + 3
    ok("consistency", run([A, A + 300000, "-k", 28]).stdout == run([A, A + 300000, "-k", 28, "-t", 1]).stdout,
       "-t 1 changed the result")
    rnd = random.Random(3)
    nums = [V71 + rnd.randrange(2 ** 64) for _ in range(40)] + ["2^1000-1", "2^71+15", "3^200"]
    for n in nums:
        out = run(["-x", n, "-k", 24], check=False)
        if "DISCARDED" in out.stdout:
            ok("consistency", out.returncode == 0 and "Checked" in out.stdout, f"-x {n}: certificate not checked")
        else:
            ok("consistency", "CANDIDATE" in out.stdout, f"-x {n}: unexpected output")
        if isinstance(n, int):
            g = generated(n, n + 1, 24)
            ok("consistency", (g == [n]) == ("CANDIDATE" in out.stdout), f"-x {n} disagrees with the generator")


def t_cli(long):
    for args in [["-k", 0], ["-k", 99], ["-l", 30], ["2^71", "abc"], ["-c", "2^71"], ["2^128"],
                 ["--nothing"], ["1", "2", "3"], ["-x", "0"], ["-q", "-1"], ["2^71+5", "2^71"],
                 ["--witnesses", "100", "50"], ["2^127"]]:
        p = run(args, check=False)
        ok("cli", p.returncode != 0 and p.stderr, f"{args}: should fail")
    p = run(["-h"], check=False)
    ok("cli", p.returncode == 0 and "Usage:" in p.stdout, "-h")
    g = [int(x) for x in run(["2^71+10^6", "2_361_183_241_434_823_606_848+1000", "-k", 20]).stdout.split()]
    ok("cli", all(V71 + 10 ** 6 <= n < V71 + 10 ** 6 + 1000 for n in g) and g, "expressions in A and B")


# --------------------------------------------------------------------------------------
# Tested/: collatz records the verified numbers; candidates does not generate them again

COLLATZ = os.path.join(ROOT, "collatz")
import tested_fmt as tfmt  # noqa: E402


def _env(d):
    return dict(os.environ, COLLATZ_TESTED=d)


def _tdir(name):
    d = os.path.join(_TESTED_TMP, name)
    _shutil.rmtree(d, True)
    return d


def _bin(d):
    f = os.path.join(d, "tested.bin")
    return tfmt.read_bin(f) if os.path.exists(f) else []


def _journal(d):
    return tfmt.read_journals(d)             # the shared one and the per-process ones


def _collatz(args, d, data=None, timeout=600):
    return subprocess.run([COLLATZ] + [str(a) for a in args], input=data, capture_output=True,
                          text=True, env=_env(d), timeout=timeout)


def _cand(args, d):
    return subprocess.run([BIN] + [str(a) for a in args], capture_output=True, text=True, env=_env(d), timeout=600)


def _ints(p):
    return [int(x) for x in p.stdout.split()]


def data_ref(n):
    """Step-by-step simulation of the --data row (map x/2, 3x+1)."""
    x, i, odd = n, 0, 0
    k = ok = j = oj = e = None
    mx, m, om = n, 0, 0
    while x != 1:
        if j is None and x & (x - 1) == 0:
            j, oj, e = i, odd, x.bit_length() - 1
        if x % 2:
            odd += 1
            x = 3 * x + 1
        else:
            x //= 2
        i += 1
        if k is None and x < n:
            k, ok = i, odd
        if x > mx:
            mx, m, om = x, i, odd
    return f"{n},{k},{ok},{j - k},{oj - ok},{e},{m},{mx},{om}"


def t_tested(long):
    # 1. only NO in [2^71, 2^127) is stored; in the end everything is in tested.bin, sorted
    d = _tdir("range")
    _collatz(["-r", 2 ** 127 - 1, V71 + 3, 27, 2 ** 127, 2 ** 200 + 1, V71 - 1, V71 + 3], d)
    ok("tested", _bin(d) == [V71 + 3, 2 ** 127 - 1] and _journal(d) == [],
       f"range: bin {_bin(d)}, journal {_journal(d)}")
    _collatz(["-r", "-s", 2 ** 100 - 1], d)                   # NOTMIN: not a complete proof
    _collatz(["-r", "-n", 3, V71 + 27], d)                     # UNKNOWN
    _collatz(["-r", "--no-record", V71 + 7], d)               # recording disabled
    _collatz(["-r", 0, -5, "abc"], d)                          # errors
    ok("tested", _bin(d) == [V71 + 3, 2 ** 127 - 1], f"NOTMIN/UNKNOWN/--no-record/errors were recorded: {_bin(d)}")
    _collatz(["-r", "-s", V71 + 27], d)                        # -s with x_σ < 2^71: NO
    ok("tested", _bin(d) == [V71 + 3, V71 + 27, 2 ** 127 - 1], "-s with NO should be recorded (in order)")

    # 2. in bulk and out of order: every NO goes, sorted and deduplicated, into the .bin
    d = _tdir("bulk")
    A, B = V71 + 10 ** 7, V71 + 10 ** 7 + 3 * 10 ** 6
    cands = _ints(_cand([A, B], d))
    half = cands[: len(cands) // 2]
    shuffled = half[:]
    random.Random(5).shuffle(shuffled)
    out = _collatz(["-r", "-s"], d, data="".join(f"{n}\n" for n in shuffled + shuffled[:50])).stdout.splitlines()
    no_nums = sorted({n for n, l in zip(shuffled, out) if is_no(l)})
    ok("tested", len(half) > 3000 and _bin(d) == no_nums and _journal(d) == [],
       f"bulk: {len(_bin(d))} in the .bin, {len(no_nums)} NO")

    # 3. the generator does not generate the tested numbers; -c, -x, -q and --tested agree
    again = _ints(_cand([A, B], d))
    ok("tested", again == [n for n in cands if n not in set(no_nums)], f"generator repeated tested numbers ({len(again)} printed)")
    ok("tested", int(_cand(["-c", A, B], d).stdout) == len(again), "--count")
    big = int(_cand(["-c", A - 2 ** 33, B + 2 ** 33], d).stdout)
    big0 = int(_cand(["-c", A - 2 ** 33, B + 2 ** 33], _tdir("empty")).stdout)
    ok("tested", big == big0 - len(no_nums), f"--count with whole blocks: {big} != {big0} − {len(no_nums)}")
    ok("tested", "ALREADY TESTED" in _cand(["-x", no_nums[0]], d).stdout, "-x of a tested number")
    ok("tested", "ALREADY TESTED" not in _cand(["-x", again[0]], d).stdout, "-x of an untested number")
    ok("tested", _ints(_cand(["-q", 5, A], d)) == again[:5], "-q should skip the tested numbers")
    ok("tested", _ints(_cand(["--tested"], d)) == no_nums, "--tested should list everything, in order")
    mid = no_nums[len(no_nums) // 2]
    ok("tested", _ints(_cand(["--tested", mid, B], d)) == [n for n in no_nums if n >= mid], "--tested A B")
    ok("tested", int(_cand(["--tested", "-c"], d).stdout) == len(no_nums), "--tested -c")

    # 4. several blocks: more than 4096 numbers, and a 2nd run interleaved with the 1st
    d = _tdir("blocks")
    _collatz(["-r", "-s"], d, data="".join(f"{n}\n" for n in cands[0::2]))
    _collatz(["-r", "-s"], d, data="".join(f"{n}\n" for n in cands[1::2]))
    ok("tested", _bin(d) == sorted(cands) and len(cands) > 4096, f"interleaving: {len(_bin(d))} of {len(cands)}")

    # 5. journal with a truncated record, a wrong CRC and garbage: only intact records count;
    #    the next recording repairs it
    d = _tdir("journal")
    os.makedirs(d)
    good = cands[:3]
    badcrc = bytearray(tfmt.record(cands[3]))
    badcrc[17] ^= 0xFF
    with open(os.path.join(d, "tested.journal"), "wb") as fh:
        fh.write(b"".join(tfmt.record(n) for n in good) + bytes(badcrc) + tfmt.record(cands[4])[:9])
    got = _ints(_cand(["-q", 4, A], d))
    ok("tested", got == cands[3:7], f"a corrupt/truncated record must not exclude anything: {got}")
    p = _cand(["-q", 1, A], d)
    ok("tested", "invalid" in p.stderr, f"should warn about the invalid record: {p.stderr.strip()}")
    _collatz(["-r", "-s", cands[10]], d)
    ok("tested", _bin(d) == good + [cands[10]] and _journal(d) == [], f"journal repair: {_bin(d)}")

    # 6. corrupt tested.bin: the generator ignores the block (and warns); collatz deletes nothing
    d = _tdir("corrupt")
    _collatz(["-r", "-s"], d, data="".join(f"{n}\n" for n in cands[:5000]))
    f = os.path.join(d, "tested.bin")
    raw = bytearray(open(f, "rb").read())
    raw[100] ^= 0x01                                       # inside the 1st block
    open(f, "wb").write(raw)
    p = _cand(["-q", 3, A], d)
    ok("tested", _ints(p)[:1] == cands[:1] and "corrupt" in p.stderr,
       f"a corrupt block should be ignored with a warning: {_ints(p)} / {p.stderr.strip()}")
    pc = _collatz(["-r", "-s", cands[6000]], d)
    ok("tested", open(f, "rb").read() == bytes(raw) and cands[6000] in _journal(d) and "corrupt" in pc.stderr,
       "with the .bin corrupt, collatz should keep everything and use only the journal")

    # 7. several collatz processes at once: nothing lost or mixed up
    d = _tdir("concurrency")
    parts = [half[i::4] for i in range(4)]
    procs = [subprocess.Popen([COLLATZ, "-r", "-s"], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, env=_env(d), text=True) for _ in parts]
    for pr, part in zip(procs, parts):
        pr.stdin.write("".join(f"{n}\n" for n in part))
        pr.stdin.close()
    for pr in procs:
        pr.wait()
    ok("tested", sorted(set(_bin(d)) | set(_journal(d))) == sorted(half) and _bin(d) == sorted(_bin(d)),
       f"concurrency: {len(_bin(d))} in the .bin + {len(_journal(d))} in the journal, expected {len(half)}")

    # 8. interrupted (Ctrl+C, kill, closed terminal) or with the output closed: stops at a safe
    #    point, with nothing half done, everything already sorted in the .bin, and exits through
    #    the same signal
    import signal
    big = "".join(f"{n}\n" for n in cands) * 700                # ~6 million lines
    inp = os.path.join(_TESTED_TMP, "big_input.txt")
    with open(inp, "w") as fh:
        fh.write(big)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        for when in (0.15, 0.4):
            d = _tdir(f"signal{int(sig)}_{when}")
            outf = os.path.join(_TESTED_TMP, f"output{int(sig)}.txt")
            with open(inp) as fi, open(outf, "w") as fo:
                pr = subprocess.Popen([COLLATZ, "-r", "-s", "--data"], stdin=fi, stdout=fo,
                                      stderr=subprocess.PIPE, env=_env(d))
                time.sleep(when)
                pr.send_signal(sig)
                err = pr.communicate()[1].decode()
            txt = open(outf).read()
            lines = txt.split("\n")
            complete = txt.endswith("\n") or txt == ""
            complete = complete and all(l.startswith("NO sigma=") and l.split()[-1].startswith("time=")
                                        for l in lines[:-1])
            nums = big.split("\n")
            printed = sorted({int(nums[i]) for i in range(len(lines) - 1)})
            data_rows = [int(x.split(",")[0]) for x in open(os.path.join(d, "data.csv")).read().splitlines()[1:]] \
                if os.path.exists(os.path.join(d, "data.csv")) else []
            leftovers = [x for x in os.listdir(d) if x.endswith(".tmp")] if os.path.exists(d) else []
            ok("tested", pr.returncode == -sig and complete and _bin(d) == printed and _journal(d) == []
               and data_rows == printed and not leftovers and "interrupted" in err,
               f"signal {sig} at {when}s: exit code {pr.returncode}, complete lines {complete}, "
               f".bin {len(_bin(d))} vs printed {len(printed)}, journal {len(_journal(d))}, "
               f"data {len(data_rows)}, leftovers {leftovers}")
    # waiting for input (pipe open, no data): stops immediately
    d = _tdir("waiting")
    pr = subprocess.Popen([COLLATZ, "-r", "-s"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, env=_env(d))
    pr.stdin.write("".join(f"{n}\n" for n in half[:10]).encode())
    pr.stdin.flush()
    time.sleep(0.5)
    t0 = time.time()
    pr.send_signal(signal.SIGINT)
    out = pr.communicate(timeout=20)[0].decode()
    ok("tested", pr.returncode == -signal.SIGINT and time.time() - t0 < 2 and out.count("\n") == 10
       and _bin(d) == sorted(half[:10]), f"Ctrl+C while waiting for input: exit code {pr.returncode}, {_bin(d)[:2]}")
    # in the middle of a huge number: that number is abandoned (nothing printed or recorded)
    d = _tdir("huge")
    pr = subprocess.Popen([COLLATZ, "-r", "2^30000000-1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_env(d))
    time.sleep(0.5)
    t0 = time.time()
    pr.send_signal(signal.SIGINT)
    out = pr.communicate(timeout=60)[0]
    ok("tested", pr.returncode == -signal.SIGINT and out == b"" and time.time() - t0 < 5,
       f"Ctrl+C on a huge number: exit code {pr.returncode}, output {out[:40]!r}, {time.time() - t0:.1f}s")
    # two signals in a row: exits immediately; what was stored is intact and holds only proven numbers
    d = _tdir("two")
    with open(inp) as fi:
        pr = subprocess.Popen([COLLATZ, "-r", "-s"], stdin=fi, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, env=_env(d))
        time.sleep(0.3)
        pr.send_signal(signal.SIGINT)
        pr.send_signal(signal.SIGINT)
        pr.wait()
    st = set(_bin(d)) | set(_journal(d))
    ok("tested", pr.returncode == -signal.SIGINT and st <= set(cands), f"two Ctrl+C: exit code {pr.returncode}")
    _collatz(["-r", 27], d)                       # the next run puts the leftovers in order
    ok("tested", set(_bin(d)) == st and _journal(d) == [], "the next run should merge the journal")
    # closed output (| head): stops at a safe point, and what was proven ends up sorted in the .bin
    d = _tdir("pipe")
    p = subprocess.run(f"{COLLATZ} -r -s < {inp} | head -1", shell=True, capture_output=True, text=True, env=_env(d))
    ok("tested", 0 < len(_bin(d)) and set(_bin(d)) <= set(cands) and _journal(d) == [] and p.stdout.count("\n") == 1,
       f"| head: {len(_bin(d))} in the .bin, journal {len(_journal(d))}")
    # candidates also stops between two lines (output to a pipe that is read slowly)
    for args in (["2^71", "-k", 20], ["2^71", "-a", 10 ** 6, "2^90", "-k", 20]):
        pr = subprocess.Popen([BIN] + [str(a) for a in args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              env=_env(_tdir("empty2")))
        first = pr.stdout.read(1 << 16) if args[1] != "-a" else b""
        time.sleep(0.3)
        pr.send_signal(signal.SIGINT)
        rest, err = pr.communicate()
        txt = (first + rest).decode()
        ok("tested", (txt == "" or txt.endswith("\n")) and all(l.isdigit() for l in txt.split("\n")[:-1])
           and pr.returncode == -signal.SIGINT, f"candidates {args} interrupted: exit code {pr.returncode}, {len(txt)} bytes")

    # --- findings of the adversarial review ---
    import fcntl
    # 2nd Ctrl+C while waiting for another process's lock: nothing is lost
    d = _tdir("lock")
    os.makedirs(d)
    lockf = open(os.path.join(d, ".lock"), "w")
    fcntl.flock(lockf, fcntl.LOCK_EX)                      # "another process" tidying up
    outf = os.path.join(_TESTED_TMP, "output_lock.txt")
    with open(inp) as fi, open(outf, "w") as fo:
        pr = subprocess.Popen([COLLATZ, "-r", "-s"], stdin=fi, stdout=fo, stderr=subprocess.PIPE, env=_env(d))
        time.sleep(0.5)
        pr.send_signal(signal.SIGINT)                      # 1st: stops and waits for the lock
        time.sleep(1.2)
        alive_after_1 = pr.poll() is None
        pr.send_signal(signal.SIGINT)                      # 2nd: exits immediately
        err = pr.communicate(timeout=20)[1].decode()
    fcntl.flock(lockf, fcntl.LOCK_UN)
    lockf.close()
    lines = open(outf).read().split("\n")
    nums = big.split("\n")
    printed = {int(nums[i]) for i in range(len(lines) - 1) if is_no(lines[i])}
    st = set(_bin(d)) | set(_journal(d))
    ok("tested", alive_after_1 and "waiting" in err and pr.returncode == -signal.SIGINT
       and printed <= st and st <= set(cands) and lines[-1] == "",
       f"2nd Ctrl+C while waiting for the lock: exit code {pr.returncode}, printed {len(printed)}, stored {len(st)}, "
       f"lost {len(printed - st)}")
    _collatz(["-r", 27], d)                                # the next run puts everything in order
    ok("tested", set(_bin(d)) == st and _journal(d) == [], "after the lock, the next run should put everything in order")
    # Ctrl+C during the final tidy-up of a normal run: exits through the signal (so script loops
    # stop). "Another process" holds the lock so the tidy-up takes as long as needed.
    d = _tdir("normal_end")
    os.makedirs(d)
    lockf = open(os.path.join(d, ".lock"), "w")
    fcntl.flock(lockf, fcntl.LOCK_EX)
    pr = subprocess.Popen([COLLATZ, "-r", str(cands[0])], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_env(d))
    time.sleep(0.8)                                        # finished the number, waiting for the lock
    waiting = pr.poll() is None
    pr.send_signal(signal.SIGINT)
    time.sleep(0.2)
    fcntl.flock(lockf, fcntl.LOCK_UN)
    lockf.close()
    out, err = pr.communicate(timeout=60)
    ok("tested", waiting and pr.returncode == -signal.SIGINT and is_no(out.decode())
       and _bin(d) == [cands[0]] and _journal(d) == [],
       f"Ctrl+C during the final tidy-up: exit code {pr.returncode}, .bin {_bin(d)}, journal {_journal(d)}")
    # huge number and huge expression: the 1st Ctrl+C exits in under 1 s, with no output at all
    for expr in ("2^60000000-1", "3^(4*10^8)+2"):
        d = _tdir("huge2")
        pr = subprocess.Popen([COLLATZ, "-r", "-s", "--", expr], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_env(d))
        time.sleep(1.0)
        t0 = time.time()
        pr.send_signal(signal.SIGINT)
        out = pr.communicate(timeout=60)[0]
        ok("tested", pr.returncode == -signal.SIGINT and out == b"" and time.time() - t0 < 1.0,
           f"Ctrl+C on {expr}: exit code {pr.returncode}, {time.time() - t0:.2f}s")
    # half an input line (producer stalled): SIGTERM exits immediately
    d = _tdir("half_line")
    pr = subprocess.Popen([COLLATZ, "-r", "-s"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_env(d))
    pr.stdin.write(f"{cands[0]}\n{str(cands[1])[:10]}".encode())
    pr.stdin.flush()
    time.sleep(0.5)
    t0 = time.time()
    pr.send_signal(signal.SIGTERM)
    out = pr.communicate(timeout=20)[0].decode()
    ok("tested", pr.returncode == -signal.SIGTERM and time.time() - t0 < 1.0 and out.count("\n") == 1
       and _bin(d) == [cands[0]], f"half a line on input: exit code {pr.returncode}, {time.time() - t0:.2f}s")
    # recording impossible: the message does not claim that anything was stored
    blocker = os.path.join(_TESTED_TMP, "not_a_dir")
    open(blocker, "w").close()
    p = subprocess.Popen([COLLATZ, "-r", "-s"], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                         env=_env(os.path.join(blocker, "sub")))
    p.stdin.write("".join(f"{n}\n" for n in half[:50]).encode())
    p.stdin.flush()
    time.sleep(1.5)
    p.send_signal(signal.SIGINT)
    err = p.communicate(timeout=20)[1].decode()
    ok("tested", "NOT" in err and "saved in" not in err, f"recording failure: {err.strip()[-160:]}")
    # | head: both programs exit with the SIGPIPE code (141)
    for prog in (f"{BIN} -k 20 2^71 2^80", f"{COLLATZ} -r -s < {inp}"):
        p = subprocess.run(["bash", "-c", f"{prog} | head -1 >/dev/null; echo ${{PIPESTATUS[0]}}"],
                           capture_output=True, text=True, env=_env(_tdir("head")))
        ok("tested", p.stdout.strip() == "141", f"{prog.split()[0]} | head: exit code {p.stdout.strip()}")
    # --witnesses interrupted: ends with a complete line
    pr = subprocess.Popen([BIN, "--witnesses", "2^71", "2^71+2^24"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          env=_env(_tdir("empty3")))
    first = pr.stdout.read(1 << 16)
    pr.send_signal(signal.SIGINT)
    rest, _ = pr.communicate(timeout=60)
    txt = (first + rest).decode()
    ok("tested", txt.endswith("\n") and pr.returncode == -signal.SIGINT
       and all(l.split()[1] in ("C", "E") for l in txt.split("\n")[:-1]), f"--witnesses interrupted: {pr.returncode}")

    # 9. migration of the old (text) tested.txt: everything goes into the .bin, and the .txt is renamed
    d = _tdir("migration")
    os.makedirs(d)
    with open(os.path.join(d, "tested.txt"), "w") as fh:
        fh.write("# header\n" + "".join(f"{n}\n" for n in reversed(cands[:100])) + f"{cands[0]}\nabc\n{cands[200]}")
    ok("tested", _ints(_cand(["-q", 2, A], d)) == cands[100:102], "before the migration the .txt should already count")
    _collatz(["-r", 27], d)
    ok("tested", _bin(d) == cands[:100] and not os.path.exists(os.path.join(d, "tested.txt"))
       and os.path.exists(os.path.join(d, "tested.txt.imported")), f"migration: {len(_bin(d))} in the .bin")

    # 10. --data: data.csv sorted by n, without repeats, checked step by step
    d = _tdir("data")
    sample = [2 ** 100, 2 ** 127 - 1, 2 ** 126 + 1, V71 + 3] + random.Random(9).sample(half, 300)
    run1, run2 = sample[:154], sample[104:]                # 50 repeated between the two
    o1 = _collatz(["-r", "--data"], d, data="".join(f"{n}\n" for n in run1)).stdout.splitlines()
    o2 = _collatz(["-r", "-s", "--data"], d, data="".join(f"{n}\n" for n in run2)).stdout.splitlines()
    no_d = {n for n, l in zip(run1, o1) if is_no(l)} | {n for n, l in zip(run2, o2) if is_no(l)}
    rows = open(os.path.join(d, "data.csv")).read().splitlines()
    want = [data_ref(n) for n in sorted(no_d)]
    ok("tested", len(no_d) > 300 and rows[0].startswith("n,k,odd_k") and rows[1:] == want,
       f"data.csv: {len(rows) - 1} rows, expected {len(want)}; 1st difference: "
       f"{next((a, b) for a, b in zip(rows[1:], want) if a != b) if rows[1:] != want else None}")
    left = [nm for nm in os.listdir(d) if nm.startswith("data.journal") and os.path.getsize(os.path.join(d, nm))]
    ok("tested", not left, f"the data journals should be emptied: {left}")
    _collatz(["-r", V71 + 99], d)              # without --data: data.csv is left untouched
    ok("tested", open(os.path.join(d, "data.csv")).read().splitlines() == rows, "without --data the csv must not change")
    # tools/data_to_sqlite.py: same rows in ascending n, journal lines included, invalid lines skipped
    extra = next(n for n in half if n not in no_d)
    with open(os.path.join(d, "data.journal.4242.csv"), "w") as fh:
        fh.write(data_ref(extra) + "\n" + data_ref(extra)[:15])           # one complete line, one truncated
    conv = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "data_to_sqlite.py"), d],
                          capture_output=True, text=True)
    want_db = sorted(rows[1:] + [data_ref(extra)], key=lambda l: int(l.split(",")[0]))
    got_db, delta_ok = [], conv.returncode == 0
    if delta_ok:
        db = sqlite3.connect(os.path.join(d, "data.sqlite"))
        for n, bits, k, okk, jk, ojk, e, tot, m, xm, xmb, om in db.execute("SELECT * FROM data ORDER BY rowid"):
            got_db.append(f"{n},{k},{okk},{jk},{ojk},{e},{m},{xm},{om}")
            x, steps = int(n), 0
            while x != 1:
                x, steps = (3 * x + 1 if x % 2 else x // 2), steps + 1
            delta_ok &= tot == steps and bits == int(n).bit_length() and xmb == int(xm).bit_length()
        db.close()
    ok("tested", got_db == want_db and delta_ok and "1 invalid line" in conv.stderr,
       f"data_to_sqlite: rc={conv.returncode}, {len(got_db)} rows (expected {len(want_db)}), "
       f"δ/bits ok={delta_ok}, stderr={conv.stderr.strip()!r}")

    # 11. without COLLATZ_TESTED, the directory is Tested/ next to the executable
    d = _tdir("binaries")
    os.makedirs(d)
    for b in (COLLATZ, BIN):
        _shutil.copy2(b, d)
    env = {k: v for k, v in os.environ.items() if k != "COLLATZ_TESTED"}
    subprocess.run([os.path.join(d, "collatz"), "-r", str(cands[0])], capture_output=True, env=env, cwd=HERE)
    ok("tested", _bin(os.path.join(d, "Tested")) == [cands[0]], "default directory")
    out = subprocess.run([os.path.join(d, "candidates"), "-q", "1", str(A)], capture_output=True, text=True,
                         env=env, cwd=HERE).stdout.split()
    ok("tested", out == [str(cands[1])], f"default directory in candidates: {out}")


# --------------------------------------------------------------------------------------
# -a N: uniform random sampling of N (untested) candidates of the interval

def _chi2_ok(counts, expected, df):
    """Chi-square with a margin of ~4.5 standard deviations (false alarm ~ 1 in 10^5)."""
    x = sum((c - expected) ** 2 / expected for c in counts)
    return x <= df + 4.5 * (2 * df) ** 0.5, x


def _deciles(sample, allc):
    pos = {n: i for i, n in enumerate(allc)}
    counts = [0] * 10
    for n in sample:
        counts[pos[n] * 10 // len(allc)] += 1
    return counts


def t_random(long):
    d0 = _tdir("rand_empty")
    # 1. properties: N numbers, distinct, sorted, in the interval, all candidates
    for K, A, B, N in [(32, V71 + 5 * 10 ** 8, V71 + 5 * 10 ** 8 + 4 * 10 ** 7, 500),
                       (20, V71 + 3, V71 + 3 + 2 * 10 ** 7, 4000),         # many 2^20 blocks
                       (32, 2 ** 126, 2 ** 127, 300),                      # huge interval
                       (16, V71 + 777, V71 + 777 + 3 * 10 ** 6, 20000)]:   # close to the total: selection
        got = _ints(_cand(["-a", N, A, B, "-k", K], d0))
        if B - A > 2 ** 40:                                                # no full list: check with -x
            members = all("CANDIDATE" in _cand(["-x", n, "-k", K], d0).stdout for n in got[:20])
            want = N
        else:
            allc = _ints(_cand([A, B, "-k", K], d0))
            members = set(got) <= set(allc)
            want = min(N, len(allc))
        ok("random", len(got) == want and got == sorted(set(got)) and all(A <= n < B for n in got) and members,
           f"K={K} [{A},{B}) N={N}: {len(got)} printed (expected {want}), members {members}")
    # 2. reproducible with --seed; different seeds, different draws
    A, B = V71 + 10 ** 9, V71 + 10 ** 9 + 10 ** 8
    a1 = _ints(_cand(["-a", 50, A, B, "--seed", 7], d0))
    a2 = _ints(_cand(["-a", 50, A, B, "--seed", 7], d0))
    a3 = _ints(_cand(["-a", 50, A, B, "--seed", 8], d0))
    ok("random", a1 == a2 and a1 != a3 and len(a1) == 50, "--seed should repeat the draw")
    p = _cand(["-v", "-a", 5, A, B], d0)
    ok("random", "seed" in p.stderr, "-v should show the seed")
    # 3. N ≥ total: all are printed (with a warning if N > total)
    A, B = V71 + 10 ** 6, V71 + 10 ** 6 + 30000
    allc = _ints(_cand([A, B], d0))
    p = _cand(["-a", len(allc) + 5, A, B], d0)
    ok("random", _ints(p) == allc and "has only" in p.stderr, "N > total should give all, with a warning")
    p = _cand(["-a", len(allc), A, B], d0)
    ok("random", _ints(p) == allc and "has only" not in p.stderr, "N = total should give all, without a warning")
    # 4. uniformity (chi-square): rejection, selection, and the frequency of each candidate
    A, B = V71 + 2 ** 40, V71 + 2 ** 40 + 4 * 10 ** 8                         # ~1.1 million candidates
    allc = _ints(_cand([A, B, "-k", 20], d0))
    for N, mode in [(30000, "rejection"), (200000, "reservoir")]:
        p = _cand(["-v", "-a", N, A, B, "-k", 20, "--seed", 123], d0)
        got = _ints(p)
        good, x = _chi2_ok(_deciles(got, allc), N / 10, 9)
        ok("random", len(got) == N and set(got) <= set(allc) and good and mode in p.stderr,
           f"{mode}: chi² of the deciles = {x:.1f}; {p.stderr.strip()[-80:]}")
    A, B = V71 + 3 * 10 ** 6, V71 + 3 * 10 ** 6 + 8000                        # ~150 candidates (K=12)
    opts = ["-k", 12, "-l", 6, "-j", 3]
    allc = _ints(_cand([A, B] + opts, d0))
    runs = 300 if not long else 1500
    for N in (4, 30):                                                         # rejection and selection
        freq = {n: 0 for n in allc}
        for i in range(runs):
            for n in _ints(_cand(["-a", N, A, B, "--seed", 1000 + i] + opts, d0)):
                freq[n] += 1
        good, x = _chi2_ok(list(freq.values()), runs * N / len(allc), len(allc) - 1)
        ok("random", good and sum(freq.values()) == runs * N, f"frequencies (N={N}, {len(allc)} candidates): chi² = {x:.0f}")
    # 5. heavily tested interval: draws only among the untested numbers, uniformly
    d = _tdir("rand_tested")
    os.makedirs(d)
    A, B = V71 + 7 * 10 ** 7, V71 + 7 * 10 ** 7 + 10 ** 8
    allc = _ints(_cand([A, B], d0))
    rnd = random.Random(3)
    tested = set(n for n in allc if rnd.random() < 0.9)
    with open(os.path.join(d, "tested.journal"), "wb") as fh:
        fh.write(b"".join(tfmt.record(n) for n in sorted(tested)))
    free = [n for n in allc if n not in tested]
    for N in (len(free) // 20, len(free) // 3):
        got = _ints(_cand(["-a", N, A, B, "--seed", 5], d))
        good, x = _chi2_ok(_deciles(got, free), N / 10, 9)
        ok("random", len(got) == N and set(got) <= set(free) and good,
           f"interval 90% tested, N={N}: {len(got)} printed, {len(set(got) - set(free))} outside the free ones, chi² {x:.1f}")
    # 7. interval almost entirely tested (rejection would yield almost nothing): the exact mode handles it
    d = _tdir("rand_almost")
    os.makedirs(d)
    A, B = V71 + 5 * 10 ** 7, V71 + 5 * 10 ** 7 + 10 ** 8
    opts = ["-k", 12, "-l", 6, "-j", 3]
    allc = _ints(_cand([A, B] + opts, d0))
    free = set(random.Random(11).sample(allc, 200))
    with open(os.path.join(d, "tested.journal"), "wb") as fh:
        fh.write(b"".join(tfmt.record(n) for n in allc if n not in free))
    for seed in range(5):
        got = _ints(_cand(["-a", 10, A, B, "--seed", seed] + opts, d))
        ok("random", len(got) == 10 and set(got) <= free, f"almost entirely tested (seed {seed}): {len(got)} printed")
    # 8. another process recording during the exact draw: never an error, always N untested numbers
    d = _tdir("rand_race")
    os.makedirs(d)
    with open(os.path.join(d, "tested.journal"), "wb") as fh:
        fh.write(b"".join(tfmt.record(n) for n in allc if n not in free))
    import threading
    def writer():
        for n in sorted(free)[:100]:
            with open(os.path.join(d, "tested.journal.99999999"), "ab") as fh:
                fh.write(tfmt.record(n))
            time.sleep(0.002)
    t = threading.Thread(target=writer)
    t.start()
    p = _cand(["-a", 50, A, B] + opts, d)
    t.join()
    got = _ints(p)
    ok("random", p.returncode == 0 and len(got) == 50 and set(got) <= free and "ERROR" not in p.stderr,
       f"draw with concurrent recording: exit code {p.returncode}, {len(got)} printed, {p.stderr.strip()[-80:]}")
    # 6. errors
    A, B = V71, V71 + 10 ** 6
    for args in (["-a", 5, A], ["-a", 0, A, B], ["-a", 5, "-c", A, B], ["-a", 5, "-q", 3, A, B],
                 ["-a", -3, A, B], ["-a", 5, A, B, "--seed", "x"], ["-a", 5, A, B, "--seed", " -1"],
                 ["-a", 5, A, B, "-x", A], ["-a", "2^64", A, B]):
        p = _cand(args, d0)
        ok("random", p.returncode != 0 and p.stderr, f"{args}: should fail")


CATS = [("certificates", t_certificates), ("tables", t_tables), ("known", t_known),
        ("generation", t_generation), ("small", t_small), ("consistency", t_consistency), ("cli", t_cli),
        ("tested", t_tested), ("random", t_random)]


def main():
    ap = argparse.ArgumentParser(description="Tests of the candidates generator.")
    ap.add_argument("--long", action="store_true", help="larger windows and larger K")
    ap.add_argument("--only", default="", help="comma-separated groups to run (default: all)")
    ap.add_argument("--bin", default=BIN, help="path of the candidates binary")
    args = ap.parse_args()
    globals()["BIN"] = os.path.abspath(args.bin)
    only = set(filter(None, args.only.split(",")))
    t0 = time.time()
    for name, fn in CATS:
        if only and name not in only:
            continue
        t1 = time.time()
        try:
            fn(args.long)
        except Exception as e:  # noqa: BLE001
            ok(name, False, f"exception: {e!r}")
        c = COUNTS.get(name, [0, 0])
        print(f"{name:14s} {c[0]:6d} ok  {c[1]:4d} failures  ({time.time() - t1:.1f} s)", flush=True)
    if "_discards checked" in COUNTS:
        print(f"{'':14s} {COUNTS['_discards checked'][0]} certificates checked by direct simulation")
    for f in FAILS:
        print("FAIL", f)
    print(f"total: {'OK' if not FAILS else 'FAILED'} in {time.time() - t0:.1f} s")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
