#!/usr/bin/env python3
"""
Differential tests of the `collatz` verifier.

    python3 tests/test_collatz.py                  default suite (~1 min)
    python3 tests/test_collatz.py --long           more cases and larger numbers (~3-4 min)
    python3 tests/test_collatz.py --only random,expressions
    python3 tests/test_collatz.py --bin ./other/collatz --stop --timeout 60

Each line of the program's -r output (all fields except 'time') is compared with
tests/reference.py, which simulates the Collatz map one step at a time. Where Python is too
slow (10^5..10^6 bits), the program is compared with itself across the internal variants
--simple (no divide and conquer and no shortcuts), --no-blocks (exact runs only)
and --no-native (no 128-bit loop), which use different code paths.

Categories: exhaustive, known, random, structured, consistency, limits,
expressions, cli, mutation.  Exit status: 0 if everything passes, 1 on any failure.
"""

import argparse
import collections
import concurrent.futures as cf
import functools
import hashlib
import os
import random
import re
import shlex
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True          # do not leave __pycache__ inside tests/
# The programs record every verified number in Tested/; the tests use a temporary directory
# so that the real file is never touched.
import atexit as _atexit, shutil as _shutil  # noqa: E402
_TESTED_TMP = tempfile.mkdtemp(prefix="collatz_tested_")
os.environ["COLLATZ_TESTED"] = _TESTED_TMP
_atexit.register(_shutil.rmtree, _TESTED_TMP, True)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import reference as ref  # noqa: E402

V71 = ref.VERIFIED_LIMIT

# Internal variants of the program: same results through different code paths.
VARIANTS = [[], ["--simple"], ["--no-blocks"], ["--no-native"]]
# For small n the native loop does everything; these variants force GMP.
SMALL_VARIANTS = [[], ["--no-native"], ["--no-native", "--no-blocks"], ["--no-native", "--simple"]]


class Config:
    bin = os.path.join(ROOT, "collatz")
    bin_display = "./collatz"
    long = False
    jobs = os.cpu_count() or 4
    seed = 20261020
    max_examples = 3
    faildir = None
    timeout = 600          # seconds per run (1800 with --long)


CFG = Config()


# --------------------------------------------------------------------------------------
# Results

class Results:
    def __init__(self):
        self.cats = collections.OrderedDict()

    def _cat(self, cat):
        return self.cats.setdefault(cat, {"ok": 0, "fail": 0, "time": 0.0,
                                          "kinds": collections.OrderedDict(), "notes": []})

    def ok(self, cat, count=1):
        self._cat(cat)["ok"] += count

    def fail(self, cat, kind, detail):
        c = self._cat(cat)
        c["fail"] += 1
        k = c["kinds"].setdefault(kind, [0, []])
        k[0] += 1
        if len(k[1]) < CFG.max_examples:
            k[1].append(detail)

    def note(self, cat, text):
        self._cat(cat)["notes"].append(text)

    def any_failure(self):
        return any(c["fail"] for c in self.cats.values())

    def summary(self):
        print()
        print("=" * 78)
        print(f"{'category':<22}{'checks':>14}{'failures':>10}{'time':>10}")
        for name, c in self.cats.items():
            print(f"{name:<22}{c['ok'] + c['fail']:>14}{c['fail']:>10}{c['time']:>9.1f}s")
        print("=" * 78)
        for name, c in self.cats.items():
            for note in c["notes"]:
                print(f"[{name}] note: {note}")
        if NEGATIVE_TIME:
            print(f"\nwarning (not counted as a failure): {len(NEGATIVE_TIME)} line(s) with a negative 'time', "
                  f"e.g. «{NEGATIVE_TIME[0]}». On macOS, CLOCK_MONOTONIC = gettimeofday − boottime "
                  f"(1 µs resolution, subject to clock adjustments).")
        for name, c in self.cats.items():
            for kind, (count, examples) in c["kinds"].items():
                print(f"\n✘ [{name}] {kind}: {count} failure(s). Examples:")
                for e in examples:
                    print("    • " + e)
        print()
        if self.any_failure():
            total = sum(c["fail"] for c in self.cats.values())
            print(f"FAILED: {total} check(s) with a mismatch.")
        else:
            print("OK: all checks passed.")


# --------------------------------------------------------------------------------------
# Running the program

Case = collections.namedtuple("Case", "n text label")


def num_text(n):
    """Text given to the program: decimal for small values, hexadecimal for large ones."""
    return str(n) if n.bit_length() <= 256 else hex(n)


def mk(n, label=None):
    return Case(n, num_text(n), label or (str(n) if n.bit_length() <= 160 else f"<{n.bit_length()} bits>"))


def run_prog(args, lines=None):
    """Run the binary with args; lines go to standard input (one item per line)."""
    kw = dict(capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=CFG.timeout)
    if lines is None:
        kw["stdin"] = subprocess.DEVNULL
    else:
        kw["input"] = "".join(s + "\n" for s in lines)
    try:
        p = subprocess.run([CFG.bin] + list(args), **kw)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:      # return what was output before the hang
        out = e.stdout or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        return None, out, "timed out"


def run_many(jobs):
    """jobs: list of (args, lines). Run in parallel and return in the same order."""
    with cf.ThreadPoolExecutor(max_workers=CFG.jobs) as ex:
        return list(ex.map(lambda j: run_prog(*j), jobs))


def save_input(text):
    if CFG.faildir is None:
        CFG.faildir = tempfile.mkdtemp(prefix="collatz-failures-")
    path = os.path.join(CFG.faildir, hashlib.sha1(text.encode()).hexdigest()[:12] + ".txt")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(text + "\n")
    return path


def repro(flags, text):
    """Command that reproduces a case (long inputs go to a file read with -f)."""
    parts = [CFG.bin_display, "-r"] + list(flags)
    if len(text) > 160:
        return " ".join(shlex.quote(p) for p in parts + ["-f", save_input(text)])
    if text.startswith("-") and not (text[1:2].isdigit() or text[1:2] == "("):
        parts.append("--")
    return " ".join(shlex.quote(p) for p in parts + [text])


RAW_TAIL = re.compile(r" time=(-?)\d+\.\d+")
RAW_RE = re.compile(r"(NO|NOTMIN|UNKNOWN|CYCLE) sigma=(\d+|-) sigmaT=(\d+|-) bound=(\d+|-) "
                    r"total=(\d+|-) totalT=(\d+|-) bits=(\d+)")
RC_OF = {"NO": 0, "NOTMIN": 0, "UNKNOWN": 2, "CYCLE": 3, "ERROR": 1}


def rc_of(verdict, stop):
    """Expected exit status: 0 = the requested conclusion was reached. NOTMIN suffices only with -s;
    without -s (the -n limit stopped before the complete proof) it gives 2, like UNKNOWN."""
    return 2 if verdict == "NOTMIN" and not stop else RC_OF[verdict]
NEGATIVE_TIME = []     # 'time' is left out of the comparison; a negative time is only a warning


def split_line(line):
    """'<fields> time=x' -> ('<fields>', format_ok)."""
    i = line.rfind(" time=")
    if i < 0:
        return line, line.startswith("ERROR ")
    head, tail = line[:i], RAW_TAIL.fullmatch(line[i:])
    if tail and tail.group(1) and len(NEGATIVE_TIME) < 1000:
        NEGATIVE_TIME.append(line)
    return head, bool(tail) and bool(RAW_RE.fullmatch(head))


def exp_line(n, stop=False, limit=None, ev=None):
    return ref.format_line(ref.expected_line(n, stop=stop, limit=limit, ev=ev))


def compare(R, cat, cases, flags, expected, result, kind_fn=None):
    """Compare the output of ONE process (one line per case) with the expected lines."""
    rc, out, err = result
    tag = " ".join(flags) or "(default)"
    if rc is None:
        lines = out.splitlines()
        stuck = cases[min(len(lines), len(cases) - 1)]
        R.fail(cat, "timed out (hung)", f"[{tag}] the process did not finish in {CFG.timeout:.0f}s; "
                                        f"{len(lines)} of {len(cases)} lines were output; hung at {stuck.label}"
                                        f"\n        repro:    {repro(flags, stuck.text)}")
        for c, exp, line in zip(cases, expected, lines[:-1]):    # the last one may be incomplete
            if split_line(line)[0] == exp:
                R.ok(cat)
            else:
                R.fail(cat, kind_fn(c, exp, line) if kind_fn else "mismatch with the reference",
                       f"{c.label}  [{tag}]\n        expected: {exp}\n        got:      {line}"
                       f"\n        repro:    {repro(flags, c.text)}")
        return
    if rc < 0:
        R.fail(cat, "the program crashed", f"[{tag}] signal {-rc}; 1st case of the batch: {cases[0].label}; "
                                           f"stderr: {err.strip()[:200]!r}")
    lines = out.splitlines()
    all_ok = len(lines) == len(cases)
    if not all_ok:
        R.fail(cat, "number of output lines", f"[{tag}] expected {len(cases)}, got {len(lines)} "
                                              f"(rc={rc}; stderr: {err.strip()[:200]!r})")
    for c, exp, line in zip(cases, expected, lines):
        head, fmt_ok = split_line(line)
        if head == exp and fmt_ok:
            R.ok(cat)
        else:
            all_ok = False
            kind = kind_fn(c, exp, line) if kind_fn else "mismatch with the reference"
            R.fail(cat, kind, f"{c.label}  [{tag}]\n        expected: {exp}\n        got:      {line}"
                              f"\n        repro:    {repro(flags, c.text)}")
    if all_ok:   # the exit status is the maximum over the inputs (0 ok, 1 error, 2 UNKNOWN, 3 CYCLE)
        erc = max(rc_of(e.split()[0], "-s" in flags) for e in expected)
        if rc == erc:
            R.ok(cat)
        else:
            R.fail(cat, "exit status", f"[{tag}] expected {erc}, got {rc} "
                                       f"({len(cases)} inputs; 1st: {cases[0].label})")


def check_cases(R, cat, cases, evs, variants, modes=(False, True), limit=None, kind_fn=None, chunk=50000):
    """Run each variant × mode over all cases and compare with the reference."""
    exp = {stop: [exp_line(c.n, stop, limit, ev) for c, ev in zip(cases, evs)] for stop in modes}
    jobs, meta = [], []
    for v in variants:
        for stop in modes:
            flags = list(v) + (["-s"] if stop else []) + ([] if limit is None else ["-n", str(limit)])
            for i in range(0, len(cases), chunk):
                jobs.append((["-r"] + flags, [c.text for c in cases[i:i + chunk]]))
                meta.append((flags, stop, i))
    for (flags, stop, i), res in zip(meta, run_many(jobs)):
        compare(R, cat, cases[i:i + chunk], flags, exp[stop][i:i + chunk], res, kind_fn)


# --------------------------------------------------------------------------------------
# Reference in parallel (processes; the reference is pure Python)

_POOL = None


def ref_events(ns, max_steps=None, stop_at_sigma=False):
    global _POOL
    fn = functools.partial(ref.events, max_steps=max_steps, stop_at_sigma=stop_at_sigma)
    ns = list(ns)
    work = sum(n.bit_length() for n in ns[:2000]) * max(1, len(ns) // 2000)
    if CFG.jobs == 1 or len(ns) < 2 or (work < 200_000 and len(ns) < 3000):
        return [fn(n) for n in ns]
    if _POOL is None:
        _POOL = cf.ProcessPoolExecutor(max_workers=CFG.jobs)
    chunk = 1 if len(ns) <= CFG.jobs * 8 else max(1, len(ns) // (CFG.jobs * 16))
    return list(_POOL.map(fn, ns, chunksize=chunk))


# --------------------------------------------------------------------------------------
# Published data

# The task cites these δ(n); the suite checks that the reference agrees with them.
KNOWN_TOTAL_TASK = {27: 111, 97: 118, 871: 178, 63728127: 949, 670617279: 986,
                    9780657630: 1132, 75128138247: 1228, 989345275647: 1348}

# Records of the total stopping time δ(n) (C steps).
DELAY_RECORDS = {
    6171: 261, 77031: 350, 230631: 442, 410011: 448, 511935: 469, 626331: 508, 837799: 524,
    1117065: 527, 1501353: 530, 1723519: 556, 2298025: 559, 3064033: 562, 3542887: 583,
    3732423: 596, 5649499: 612, 6649279: 664, 8400511: 685, 11200681: 688, 14934241: 691,
    15733191: 704, 31466382: 705, 36791535: 744, 127456254: 950, 169941673: 953,
    226588897: 956, 268549803: 964, 537099606: 965, 1341234558: 987, 7887663552367: 1563,
    942488749153153: 1862, 7579309213675935: 1958, 93571393692802302: 2091,
    931386509544713451: 2283}

# Records of the stopping time σ(n) (C steps).
STOPPING_RECORDS = {
    3: 6, 7: 11, 27: 96, 703: 132, 10087: 171, 35655: 220, 270271: 267, 362343: 269,
    381727: 282, 626331: 287, 1027431: 298, 1126015: 365, 8088063: 401, 13421671: 468,
    20638335: 476, 26716671: 486, 56924955: 502, 63728127: 613, 217740015: 644,
    1200991791: 649, 1827397567: 706, 2788008987: 729, 12235060455: 892}

# Path records of T. Oliveira e Silva (sweet.ua.pt/tos/3x+1.html, t0.txt):
# n, s(n) = σ in T steps, t(n) = maximum of the T trajectory, m(n) = step of the maximum.
PATH_RECORDS_TOS = [
    (2, 1, 2, 0), (3, 4, 8, 2), (7, 7, 26, 3), (15, 7, 80, 4), (27, 59, 4616, 45),
    (255, 13, 6560, 8), (447, 40, 19682, 31), (639, 23, 20762, 14), (703, 81, 125252, 48),
    (1819, 61, 638468, 28), (4255, 70, 3405068, 49), (4591, 65, 4076810, 33),
    (9663, 54, 13557212, 26), (20895, 72, 25071632, 50), (26623, 65, 53179010, 35),
    (31911, 59, 60506432, 43), (60975, 127, 296639576, 67), (77671, 105, 785412368, 39),
    (113383, 110, 1241055674, 69), (138367, 59, 1399161680, 39)]
# The remaining records of Oliveira e Silva (up to 2^61) and those of D. Barina (pcbarina.fit.vutbr.cz,
# path-records.htm, n of 65 to 71 bits, maximum of T between 2^127 and 2^139).
PATH_RECORDS_MORE = [
    159487, 270271, 665215, 704511, 1042431, 1212415, 1441407, 1875711, 1988859, 2643183,
    2684647, 3041127, 3873535, 4637979, 5656191, 6416623, 6631675, 19638399, 38595583,
    80049391, 120080895, 210964383, 319804831, 1410123943, 8528817511, 12327829503,
    23035537407, 45871962271, 51739336447, 59152641055, 59436135663, 70141259775,
    77566362559, 110243094271, 204430613247, 231913730799, 272025660543, 446559217279,
    567839862631, 871673828443, 2674309547647, 3716509988199, 9016346070511,
    64848224337147, 116050121715711, 201321227677935, 265078413377535, 291732129855135,
    394491988532895, 406738920960667, 613450176662511, 737482236053119, 1254251874774375,
    5323048232813247, 8562235014026655, 10709980568908647, 49163256101584231,
    82450591202377887, 93264792503458119, 172545331199510631, 212581558780141311,
    255875336134000063, 484549993128097215, 562380758422254271, 628226286374752923,
    891563131061253151, 1038743969413717663, 1980976057694848447,
    35136221158664800255, 48503373501652785087, 55247846101001863167, 71149323674102624415,
    274133054632352106267, 1378299700343633691495, 1735519168865914451271,
    1765856170146672440559, 2358909599867980429759]


def t_max(n):
    """Maximum of the T trajectory and the step at which it occurs (only to check the table)."""
    x, best, at, i = n, n, 0, 0
    while x != 1:
        x = x // 2 if x % 2 == 0 else (3 * x + 1) // 2
        i += 1
        if x > best:
            best, at = x, i
    return best, at


# --------------------------------------------------------------------------------------
# 1. Exhaustive

def cat_exhaustive(R):
    cat = "1 exhaustive"
    top = 2_000_000 if CFG.long else 200_000
    block = 200_000
    for start in range(1, top + 1, block):
        ns = list(range(start, min(top, start + block - 1) + 1))
        evs = ref_events(ns)
        cases = [Case(n, str(n), str(n)) for n in ns]
        check_cases(R, cat, cases, evs, SMALL_VARIANTS)
    R.note(cat, f"n = 1..{top}, default and -s modes, variants " +
           ", ".join(" ".join(v) or "normal" for v in SMALL_VARIANTS))


# --------------------------------------------------------------------------------------
# 2. Known values

def cat_known(R):
    cat = "2 known"
    tables = [("δ cited in the task", KNOWN_TOTAL_TASK, "total"),
              ("δ record", DELAY_RECORDS, "total"),
              ("σ record", STOPPING_RECORDS, "sigma")]
    ns = sorted(set(KNOWN_TOTAL_TASK) | set(DELAY_RECORDS) | set(STOPPING_RECORDS)
                | {r[0] for r in PATH_RECORDS_TOS} | set(PATH_RECORDS_MORE))
    evs = dict(zip(ns, ref_events(ns)))

    # The reference itself must agree with the published values.
    for name, table, key in tables:
        for n, v in table.items():
            if evs[n][key] == v:
                R.ok(cat)
            else:
                R.fail(cat, "reference disagrees with a published value", f"{name}: n={n}: table {v}, reference {evs[n][key]}")
    for n, s, t, m in PATH_RECORDS_TOS:
        if evs[n]["sigmaT"] == s and t_max(n) == (t, m):
            R.ok(cat)
        else:
            R.fail(cat, "reference disagrees with a published value", f"path record n={n}")

    # The program: published value and complete line against the reference.
    cases = [mk(n) for n in ns]
    res = run_prog(["-r"], [c.text for c in cases])
    lines = dict(zip(ns, res[1].splitlines()))
    for name, table, key in tables + [("σ in T steps (Oliveira e Silva)", {r[0]: r[1] for r in PATH_RECORDS_TOS}, "sigmaT")]:
        for n, v in table.items():
            got = re.search(rf"\b{key}=(\S+)", lines.get(n, ""))
            if got and got.group(1) == str(v):
                R.ok(cat)
            else:
                R.fail(cat, "published value", f"{name}: n={n}: expected {key}={v}, got «{lines.get(n)}»"
                                               f"\n        repro:    {repro([], str(n))}")
    check_cases(R, cat, cases, [evs[n] for n in ns], VARIANTS)
    check_cases(R, cat, cases, [evs[n] for n in ns], [["--no-native", "--simple"], ["--no-native", "--no-blocks"]])
    R.note(cat, f"{len(KNOWN_TOTAL_TASK)} cited δ (all agree with the reference), {len(DELAY_RECORDS)} δ records, "
                f"{len(STOPPING_RECORDS)} σ records, {len(PATH_RECORDS_TOS) + len(PATH_RECORDS_MORE)} path records")


# --------------------------------------------------------------------------------------
# 3. Random

def cat_random(R):
    cat = "3 random"
    rng = random.Random(CFG.seed)
    count = 6000 if CFG.long else 600
    special = [63, 64, 65, 70, 71, 72, 127, 128, 129, 192, 193]
    ns = []
    for i in range(count):
        u = rng.random()
        if u < 0.10:
            bits = rng.choice(special)
        elif u < 0.45:
            bits = rng.randint(8, 128)
        elif u < 0.75:
            bits = rng.randint(129, 1024)
        else:
            bits = rng.randint(1025, 4096)
        n = rng.getrandbits(bits) | (1 << (bits - 1))
        if i % 7 == 1:                              # run of trailing 1s
            n |= (1 << rng.randint(1, min(bits - 1, 300))) - 1
        elif i % 7 == 2 and bits > 2:               # run of trailing 0s
            z = rng.randint(1, min(bits - 2, 300))
            n = (n >> z) << z
        ns.append(n)
    evs = ref_events(ns)
    cases = [mk(n, f"{num_text(n) if n.bit_length() <= 128 else hex(n)[:18] + '…'} ({n.bit_length()} bits)") for n in ns]
    check_cases(R, cat, cases, evs, VARIANTS + [["--no-native", "--simple"]])
    R.note(cat, f"{count} numbers of 8 to 4096 bits (seed {CFG.seed})")


# --------------------------------------------------------------------------------------
# 4. Structured inputs

def structured_cases():
    out = collections.OrderedDict()

    def add(n, label):
        if n >= 1 and n not in out:
            out[n] = label

    K = 5000
    ks = range(1, K + 1) if CFG.long else list(range(1, 401)) + list(range(401, K + 1, 41))
    for k in ks:
        p = 1 << k
        for d in (0, 1, -1, 3, -3):
            add(p + d, f"2^{k}{d:+d}" if d else f"2^{k}")
    for k in (range(1, K + 1) if CFG.long else list(range(1, 1001)) + list(range(1001, K + 1, 7))):
        add((1 << k) - 1, f"2^{k}-1")
    for k in range(1, 2001 if CFG.long else 601):
        p = 3 ** k
        for d in (0, 1, -1, 2, -2):
            add(p + d, f"3^{k}{d:+d}" if d else f"3^{k}")
    for a in range(0, 81 if CFG.long else 41):
        for b in range(0, 81 if CFG.long else 41):
            p = (1 << a) * 3 ** b
            add(p - 1, f"2^{a}*3^{b}-1")
            add(p + 1, f"2^{a}*3^{b}+1")
    for k in range(1, 121):
        for j in (1, 2, 3, 5, 31, 63, 64, 65, 100):
            add(((1 << k) - 1) << j, f"(2^{k}-1)*2^{j}")
    for k in range(1, 301):
        for m in (3, 5, 7, 9, 11):
            add((m << k) - 1, f"{m}*2^{k}-1")
    for p in (63, 64, 65, 70, 71, 72, 127, 128, 129, 192, 256):
        for d in range(-5, 6):
            add((1 << p) + d, f"2^{p}{d:+d}" if d else f"2^{p}")
    for k in range(1, 1001 if CFG.long else 301):
        add((10 ** k - 1) // 9, f"(10^{k}-1)/9")

    rng = random.Random(CFG.seed + 4)
    for i in (1, 2, 3, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 150, 191, 192, 193, 255, 256, 257, 500, 1000):
        for gap in (1, 2, 50, 63, 64, 65, 300):
            for msize in (1, 2, 10, 64, 200):
                m = rng.getrandbits(msize) | 1
                j = i + gap
                add(m * 2 ** j + (2 ** i - 1), f"{m}*2^{j}+2^{i}-1")
    for m in (1, 3, 12345, 2 ** 61 - 1, rng.getrandbits(200) | 1):
        add(m * 2 ** 200 + (2 ** 150 - 1), f"{m}*2^200+2^150-1")
        add(m * 2 ** 300, f"{m}*2^300")
    for j in (63, 64, 65, 100, 128, 200, 1000):
        for msize in (1, 3, 64, 200):
            m = rng.getrandbits(msize) | 1
            add(m << j, f"{m}*2^{j}")
    # Long runs created IN THE MIDDLE of the trajectory (not only in the bits of n): n = 2^a·m − 1 gives a
    # odd steps and x = 3^a·m − 1; with m ≡ 3^(−a) (mod 2^Z), x ends in ≥ Z zeros, and a safe block
    # starts with a long run of 0s. With m ≡ 3^(−a)·(1 − 2^z) (mod 2^(z+W)), after
    # exactly z zeros come ≥ W ones.
    for a in (1, 2, 3, 5, 10, 30, 64, 100, 200):
        for Z in (64, 65, 100, 200, 500):
            for hbits in (1, 40, 300):
                m = pow(3, -a, 2 ** Z) + (rng.getrandbits(hbits) | 1) * 2 ** Z
                add((m << a) - 1, f"2^{a}·m−1, m ≡ 3^-{a} (mod 2^{Z}), m={hex(m)[:14]}…")
        for z, W in ((1, 64), (5, 100), (64, 64), (70, 200), (200, 65)):
            mod = 2 ** (z + W)
            m = (pow(3, -a, mod) * (1 - 2 ** z)) % mod + (rng.getrandbits(50) | 1) * mod
            add((m << a) - 1, f"2^{a}·m−1 with {z} zeros and then {W} ones, m={hex(m)[:14]}…")
    for size in (100, 300, 1000, 3000):            # few bits set / almost all bits set
        for _ in range(6):
            s = sum(1 << b for b in rng.sample(range(size), rng.randint(2, 20))) | (1 << (size - 1))
            add(s, f"sparse, {size} bits: {hex(s)[:20]}…")
            add((1 << size) - 1 - (s ^ (1 << (size - 1))), f"dense, {size} bits")
    records = [r[0] for r in PATH_RECORDS_TOS] + PATH_RECORDS_MORE
    for r in records:                               # trajectories with the largest excursions
        add(r, f"path record {r}")
        add(2 * r, f"2*{r}")
        add(4 * r + 1, f"4*{r}+1")
    return [Case(n, num_text(n), label) for n, label in out.items()]


def cat_structured(R):
    cat = "4 structured"
    cases = structured_cases()
    evs = ref_events([c.n for c in cases])
    check_cases(R, cat, cases, evs, VARIANTS + [["--no-native", "--simple"], ["--no-native", "--no-blocks"]])
    R.note(cat, f"{len(cases)} distinct inputs (2^k±1, 2^k±3, 2^k−1 up to k=5000, 3^k±1, 2^a·3^b±1, "
                f"(2^k−1)·2^j, neighbours of 2^63/2^64/2^71/2^127/2^128, repunits, long runs of 1s/0s, "
                f"path records)")
    R.note(cat, "no n < 2^64 has a trajectory above 2^128 (the path record below 3.5·10^19 "
                "reaches 2^124.6), so the fallback from the native loop to GMP on overflow cannot be reached "
                "by real inputs; the 65–71-bit records exceed 2^128 inside GMP.")


# --------------------------------------------------------------------------------------
# 5. Consistency between variants on large inputs

def big_cases():
    rng = random.Random(CFG.seed + 5)
    out = []

    def rnd(bits):
        return rng.getrandbits(bits) | (1 << (bits - 1))

    sizes = [(10 ** 4, 4), (3 * 10 ** 4, 2), (10 ** 5, 3), (3 * 10 ** 5, 2), (10 ** 6, 1)]
    if CFG.long:
        sizes = [(10 ** 4, 10), (3 * 10 ** 4, 6), (10 ** 5, 6), (3 * 10 ** 5, 4), (10 ** 6, 4)]
    for bits, count in sizes:
        for i in range(count):
            out.append(Case(rnd(bits), None, f"random {bits} bits #{i + 1} (seed {CFG.seed + 5})"))
    ks = [10 ** 4, 30011, 10 ** 5] + ([3 * 10 ** 5, 10 ** 6] if CFG.long else [])
    for k in ks:
        out.append(Case((1 << k) - 1, None, f"2^{k}-1"))
        out.append(Case((1 << k) + 1, None, f"2^{k}+1"))
    for k in [6310, 63093] + ([630930] if CFG.long else []):
        out.append(Case(3 ** k, None, f"3^{k}"))
    for d in [3011, 30103]:
        out.append(Case((10 ** d - 1) // 9, None, f"(10^{d}-1)/9"))
    m = rnd(5000)
    out.append(Case(m * 2 ** 20000 + 2 ** 15000 - 1, None, f"m*2^20000+2^15000-1, m random of 5000 bits"))
    out.append(Case(m * 2 ** 60000 + 2 ** 64 - 1, None, f"m*2^60000+2^64-1"))
    for size in (10 ** 5,):
        s = sum(1 << b for b in rng.sample(range(size), 20)) | (1 << (size - 1))
        out.append(Case(s, None, f"sparse, {size} bits (20 bits set)"))
        out.append(Case((1 << size) - 1 - (s ^ (1 << (size - 1))), None, f"dense, {size} bits"))
    return [Case(c.n, hex(c.n), c.label) for c in out]


def cat_consistency(R):
    cat = "5 consistency"
    cases = big_cases()
    limits = [1000, 10007]
    # Variants × modes (× limits), one process per input: the slowest ones run in parallel.
    configs = [(stop, None) for stop in (False, True)] + [(stop, N) for stop in (False, True) for N in limits]
    jobs, meta = [], []
    for ci, c in enumerate(cases):
        for stop, N in configs:
            for v in VARIANTS:
                flags = list(v) + (["-s"] if stop else []) + ([] if N is None else ["-n", str(N)])
                if (c.n.bit_length() > 4 * 10 ** 5 and not CFG.long and v == ["--no-blocks"]
                        and not stop and N is None and not c.label.startswith("random")):
                    continue            # structured 10^6 bits without blocks: only with --long
                jobs.append((["-r"] + flags, [c.text]))
                meta.append((ci, stop, N, tuple(v), flags))
    t0 = time.time()
    results = run_many(jobs)
    R.note(cat, f"{len(cases)} inputs of 10^4 to 10^6 bits; {len(jobs)} runs in {time.time() - t0:.1f}s")

    by_key = collections.defaultdict(dict)
    for (ci, stop, N, v, flags), res in zip(meta, results):
        by_key[(ci, stop, N)][v] = (flags, res)

    # Reference where it is cheap: -s up to a cap on steps (~0.2 s of Python per input),
    # the short limits, and the full default mode up to 3·10^4 bits (10^5 with --long).
    specs = {}
    for ci, c in enumerate(cases):
        bits = c.n.bit_length()
        specs[(ci, "s")] = (c.n, (("max_steps", max(2000, int(2e9 / bits))), ("stop_at_sigma", True)))
        for N in limits:
            specs[(ci, N)] = (c.n, (("max_steps", N),))
        if bits <= (10 ** 5 if CFG.long else 3 * 10 ** 4):
            specs[(ci, "full")] = (c.n, ())
    groups = collections.defaultdict(list)
    for key, (n, kw) in specs.items():
        groups[kw].append((key, n))
    refs = {}
    for kw, items in groups.items():
        for (key, n), ev in zip(items, ref_events([n for _, n in items], **dict(kw))):
            refs[key] = ev

    def ref_for(ci, stop, N):
        if N is not None:
            return refs.get((ci, N))
        return refs.get((ci, "s" if stop else "full"))

    checked_ref = 0
    for (ci, stop, N), runs in by_key.items():
        c = cases[ci]
        heads = {}
        for v, (flags, (rc, out, err)) in runs.items():
            line = out.strip()
            head, fmt_ok = split_line(line)
            if rc is None or rc < 0 or not fmt_ok:
                R.fail(cat, "run failed", f"{c.label} [{' '.join(flags)}]: rc={rc} output «{line[:200]}» "
                                          f"stderr «{err.strip()[:200]}»\n        repro:    {repro(flags, c.text)}")
                continue
            heads[v] = (head, line, flags)
        # 1) all variants equal
        distinct = {h for h, _, _ in heads.values()}
        if len(distinct) <= 1:
            R.ok(cat)
        else:
            detail = "\n".join(f"        {(' '.join(f) or '(default)'):<28} {line}" for _, line, f in heads.values())
            R.fail(cat, "variants disagree with each other" + (" (with -n)" if N is not None else ""),
                   f"{c.label}\n{detail}\n        repro:    {repro(next(iter(heads.values()))[2], c.text)}")
        # 2) against the reference, when it was computed and covers what is needed
        ev = ref_for(ci, stop, N)
        if ev is None:
            continue
        try:
            exp = exp_line(c.n, stop, N, ev)
        except ValueError:
            continue                                   # the reference did not get far enough
        checked_ref += 1
        for v, (head, line, flags) in heads.items():
            if head == exp:
                R.ok(cat)
            else:
                kind = ("false CYCLE under --max-steps" if line.startswith("CYCLE") else
                        "mismatch with the reference" + (" (with -n)" if N is not None else ""))
                R.fail(cat, kind, f"{c.label}  [{' '.join(flags) or '(default)'}]\n        expected: {exp}"
                                  f"\n        got:      {line}\n        repro:    {repro(flags, c.text)}")
    R.note(cat, f"{checked_ref} input×mode×limit combinations also checked against the reference")


# --------------------------------------------------------------------------------------
# 6. --max-steps

def limit_sample():
    rng = random.Random(CFG.seed + 6)
    ns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 27, 31, 97, 703, 871, 6171, 77031, 837799, 63728127,
          670617279, 9780657630, 989345275647, 1980976057694848447,
          2 ** 63 - 1, 2 ** 63 + 1, 2 ** 64 - 1, 2 ** 64, 2 ** 64 + 1, 2 ** 64 + 3, 2 ** 65 - 1,
          2 ** 70 + 1, 2 ** 71 - 1, 2 ** 71, 2 ** 71 + 1, 2 ** 71 + 3, 2 ** 71 + 7, 2 ** 72 - 1,
          35136221158664800255, 2358909599867980429759, 2 ** 80 - 1, 2 ** 100 + 1, 3 ** 60,
          2 ** 127 - 1, 2 ** 127 + 1, 2 ** 128 - 1, 2 ** 128 + 1, 2 ** 128 + 3, (10 ** 40 - 1) // 9,
          2 ** 200 - 1, 12345 * 2 ** 200 + 2 ** 150 - 1, 7 * 2 ** 300]
    for bits in (90, 100, 300, 1000, 2000) + ((3000, 4000) if CFG.long else ()):
        ns.append(rng.getrandbits(bits) | (1 << (bits - 1)))
    return ns


def cat_limits(R):
    cat = "6 limits"
    rng = random.Random(CFG.seed + 66)
    ns = limit_sample()
    evs = ref_events(ns)
    limits = set()
    for n, ev in zip(ns, evs):
        for key in ("sigma", "bound", "total"):
            if ev[key] is not None:
                limits.update(ev[key] + d for d in (-1, 0, 1))
        limits.update(rng.randint(0, ev["total"] + 2) for _ in range(6 if CFG.long else 3))
    limits.update((0, 1, 2, 3, 4, 5))
    limits = sorted(N for N in limits if N >= 0)
    cases = [mk(n) for n in ns]

    def kind_for(N):
        def kind(c, exp, line):
            if line.startswith("CYCLE"):
                return "false CYCLE under --max-steps (\"period of 0 steps\")"
            if N == 0:
                return "-n 0 is treated as \"no limit\""
            return "mismatch with the reference"
        return kind

    variants = VARIANTS + ([["--no-native", "--simple"], ["--no-native", "--no-blocks"]] if CFG.long else [])
    exp = {(N, stop): [exp_line(n, stop, N, ev) for n, ev in zip(ns, evs)] for N in limits for stop in (False, True)}
    jobs, meta = [], []
    for N in limits:
        for stop in (False, True):
            for v in variants:
                flags = list(v) + (["-s"] if stop else []) + ["-n", str(N)]
                jobs.append((["-r"] + flags, [c.text for c in cases]))
                meta.append((N, stop, flags))
    for (N, stop, flags), res in zip(meta, run_many(jobs)):
        compare(R, cat, cases, flags, exp[(N, stop)], res, kind_for(N))

    # "Never more than N steps": the -r fields only show events, but the readable UNKNOWN output
    # says how many steps were taken ("steps taken: X"); X must be ≤ N.
    steps_re = re.compile(r"steps taken: ([\d,]+)")
    hr_jobs, hr_meta = [], []
    for N in limits:
        for stop in (False, True):
            idx = [i for i, e in enumerate(exp[(N, stop)]) if e.startswith("UNKNOWN")]
            for v in ([], ["--no-blocks"], ["--simple"]) if idx else ():
                flags = list(v) + (["-s"] if stop else []) + ["-n", str(N)]
                hr_jobs.append((flags, [cases[i].text for i in idx]))
                hr_meta.append((N, flags, idx))
    short = 0
    for (N, flags, idx), (rc, out, err) in zip(hr_meta, run_many(hr_jobs)):
        blocks = re.split(r"^n = ", out, flags=re.M)[1:]
        if len(blocks) != len(idx):
            R.fail(cat, "incomplete readable output", f"[{' '.join(flags)}] {len(idx)} inputs, {len(blocks)} blocks (rc={rc})")
            continue
        for i, blk in zip(idx, blocks):
            m = steps_re.search(blk)
            if m and int(m.group(1).replace(",", "")) <= N:
                R.ok(cat)
                short += int(m.group(1).replace(",", "")) < N
            else:
                kind = ("false CYCLE under --max-steps (\"period of 0 steps\")" if "COUNTEREXAMPLE" in blk else
                        "-n 0 is treated as \"no limit\"" if N == 0 else
                        "step limit exceeded (readable output)")
                R.fail(cat, kind, f"{cases[i].label}  [{' '.join(flags)}]: expected UNKNOWN with ≤ {N} steps, got "
                                  f"«{' '.join(blk.split())[:160]}»\n        repro:    "
                                  f"{repro(flags, cases[i].text).replace(' -r ', ' ', 1)}")
    R.note(cat, f"{len(ns)} values of n × {len(limits)} limits N (0..3, σ±1, bound±1, δ±1 and random) × "
                f"2 modes × {len(variants)} variants; number of steps taken checked in the readable output of UNKNOWN "
                f"({short} of them stopped at N−1: the next T step could be worth 2 C steps)")


# --------------------------------------------------------------------------------------
# 7. Expressions

def expression_table():
    """(text for the program, value from Python).  The program's division and % = Python's // and %."""
    big_sum = "+".join(["1"] * 500)
    return [
        ("27", 27), ("2^100-1", 2 ** 100 - 1), ("(10^50-1)/9", (10 ** 50 - 1) // 9), ("3**40", 3 ** 40),
        ("0x1F", 0x1F), ("0X1f", 0x1F), ("0b1011", 0b1011), ("0B1011", 0b1011),
        ("2^3^2", 2 ** 3 ** 2), ("(2^3)^2", (2 ** 3) ** 2), ("-2^2+10", -2 ** 2 + 10),
        ("2^-0", 2 ** -0), ("10-2-3", 10 - 2 - 3), ("100/10/2", 100 // 10 // 2), ("2*3^2", 2 * 3 ** 2),
        ("2^2*3", 2 ** 2 * 3), ("2+3*4^2", 2 + 3 * 4 ** 2), ("7%3", 7 % 3), ("-7%3", -7 % 3),
        ("10+(-7)%3", 10 + (-7) % 3), ("10+-8/3", 10 + -8 // 3), ("10--8/3", 10 - -8 // 3),
        ("10+5%-3", 10 + 5 % -3), ("10+7/-2", 10 + 7 // -2), ("8/3*3", 8 // 3 * 3),
        ("2^10%1000", 2 ** 10 % 1000), ("2^2^2^2", 2 ** 2 ** 2 ** 2), ("3^2^0", 3 ** 2 ** 0),
        ("2^0^0", 2 ** 0 ** 0), ("0^0", 0 ** 0), ("(-1)^4", (-1) ** 4), ("(-2)^2", (-2) ** 2),
        ("(-2)^3+10", (-2) ** 3 + 10), ("(-1)^(10^30)", 1), ("1^(10^30)", 1), ("8^(1/3)", 8 ** (1 // 3)),
        ("2^(3+4)", 2 ** (3 + 4)), ("((2))^((3))", 8), ("(((7)))", 7), ("0x10^2", 0x10 ** 2),
        ("0b11^0b11", 0b11 ** 0b11), ("0x1F+0b11", 0x1F + 0b11), ("0010", 10), ("0x0010", 16),
        ("0b0011", 3), ("1_000_000", 1_000_000), ("1 000 000", 1000000), ("  2 ** 10  ", 1024),
        ("2**3**2", 2 ** 3 ** 2), ("+5", 5), ("--5", 5), ("-(-5)", 5), ("+-+-5", 5),
        ("2^64+1", 2 ** 64 + 1), ("(2^127-1)*(2^61-1)", (2 ** 127 - 1) * (2 ** 61 - 1)),
        ("(3^100-1)/2", (3 ** 100 - 1) // 2),
        ("123456789012345678901234567890%1000000007", 123456789012345678901234567890 % 1000000007),
        ("0xDEADBEEFdeadbeef", 0xDEADBEEFDEADBEEF), ("2^71+1", 2 ** 71 + 1), ("10^30/7", 10 ** 30 // 7),
        ("(2^200-1)/(2^100+1)", (2 ** 200 - 1) // (2 ** 100 + 1)),
        ("99999999999999999999999999999999999999 % 2^64", 99999999999999999999999999999999999999 % 2 ** 64),
        ("2^1000+3^600-5^400", 2 ** 1000 + 3 ** 600 - 5 ** 400), (big_sum, 500),
        ("(" * 300 + "3" + ")" * 300, 3), ("-" * 300 + "5", 5), ("2^(2^4)", 65536),
        ("0x" + "f" * 64, 16 ** 64 - 1), ("0b1" + "0" * 99, 2 ** 99), ("2^1_0", 1024),
    ]


INVALID_EXPRESSIONS = [
    "abc", "1e5", "1.5", "2^-1", "2^(0-1)", "7/0", "7%0", "(1", "1)", "()", "5-", "*5", "^2",
    "0x", "0b", "0b102", "0xG", "2^^3", "2***3", "00x10", "2^(10^30)", "10^(10^20)", "2^(2^33+1)",
    "x", "1,5", "１２", "2^", "(2)(3)", "5 6 +", "--", "0x-1",
    "2 * * 3", "2* *3", "27 31", "10 0000", "1 00",
]
NONPOSITIVE_EXPRESSIONS = ["0", "-5", "1-1", "2-3", "0^5", "(-1)^3", "-2^2", "5%-3", "0x0", "-0", "-(2^100)"]


def cat_expressions(R):
    cat = "7 expressions"
    table = expression_table()
    cases = [Case(v, text, f"«{text if len(text) <= 60 else text[:57] + '...'}» = {v if v.bit_length() <= 128 else hex(v)[:20] + '…'}")
             for text, v in table]
    evs = ref_events([v for _, v in table])
    exp = [exp_line(v, False, None, ev) for (_, v), ev in zip(table, evs)]
    compare(R, cat, cases, [], exp, run_prog(["-r"], [c.text for c in cases]),
            lambda c, e, line: "expression value differs from Python")

    # The exact value: (expr) - (v-1) must give 1 (uses only parentheses, subtraction and decimal).
    one = exp_line(1)
    cases1 = [Case(1, f"({text})-({v - 1})", f"«({text[:40]})-(v-1)» should be 1") for text, v in table]
    compare(R, cat, cases1, [], [one] * len(cases1), run_prog(["-r"], [c.text for c in cases1]),
            lambda c, e, line: "expression value differs from Python")

    bad = [Case(0, t, f"«{t}» (invalid)") for t in INVALID_EXPRESSIONS]
    compare(R, cat, bad, [], ["ERROR invalid_input"] * len(bad), run_prog(["-r"], [c.text for c in bad]),
            lambda c, e, line: "invalid expression not rejected as expected")
    nonpos = [Case(0, t, f"«{t}» (≤ 0)") for t in NONPOSITIVE_EXPRESSIONS]
    compare(R, cat, nonpos, [], ["ERROR not_positive"] * len(nonpos), run_prog(["-r"], [c.text for c in nonpos]),
            lambda c, e, line: "value ≤ 0 not rejected as expected")

    # On the command line: numbers starting with '-' or '(' are not options.
    argv_cases = [(["-2^2+10"], exp_line(6)), (["-(-5)"], exp_line(5)), (["+5"], exp_line(5)),
                  (["(27)"], exp_line(27)), (["--", "--5"], exp_line(5)), (["-5"], "ERROR not_positive"),
                  ([""], "ERROR invalid_input"), (["   "], "ERROR invalid_input"),
                  (["#27"], "ERROR invalid_input"), (["2^100", "-1"], None)]
    for args, expected in argv_cases:
        rc, out, err = run_prog(["-r"] + args)
        if expected is None:            # two arguments = two numbers (2^100 and -1)
            want = [exp_line(2 ** 100), "ERROR not_positive"]
            got = [split_line(l)[0] for l in out.splitlines()]
            ok = got == want and rc == 1
        else:
            got = [split_line(l)[0] for l in out.splitlines()]
            ok = got == [expected]
        if ok:
            R.ok(cat)
        else:
            R.fail(cat, "command-line argument", f"collatz -r {' '.join(shlex.quote(a) for a in args)}: "
                                                 f"expected {expected!r}, got {out.strip()!r} (rc={rc})")

    with tempfile.TemporaryDirectory(prefix="collatz-test-") as tmp:
        # Robustness: a deeply nested expression may give a result or an ERROR, but must not crash the program.
        for desc, text, gen in [
                ("100000 nested parentheses", "(" * 100000 + "27" + ")" * 100000, "'(' * 100000 + '27' + ')' * 100000"),
                ("1000000 '-' signs", "-" * 1000000 + "27", "'-' * 1000000 + '27'"),
                ("500000 chained powers 1^1^...^1", "1^" * 500000 + "1", "'1^' * 500000 + '1'")]:
            path = os.path.join(tmp, "nested.txt")
            with open(path, "w") as f:
                f.write(text)
            rc, out, err = run_prog(["-r", "-f", path])
            if rc is not None and rc >= 0 and out.strip() and (out.startswith("ERROR") or split_line(out.strip())[1]):
                R.ok(cat)
            else:
                R.fail(cat, "the program crashes (stack overflow) on a deeply nested expression",
                       f"{desc}: rc={rc}{f' (signal {-rc})' if rc is not None and rc < 0 else ''}, "
                       f"output «{out.strip()[:80]}»\n        repro:    python3 -c \"print({gen})\" > /tmp/p.txt && "
                       f"{CFG.bin_display} -r -f /tmp/p.txt")

        # File (-f): spaces, line breaks and '_' inside the number/expression.
        for content, v in [("2^100\n - 1\n", 2 ** 100 - 1), ("123 456\n789_000\n", 123456789000),
                           ("(10^50\n-1)\n/9", (10 ** 50 - 1) // 9), ("0x" + "\n".join(["ff"] * 40), 2 ** 320 - 1)]:
            path = os.path.join(tmp, f"n{v.bit_length()}.txt")
            with open(path, "w") as f:
                f.write(content)
            rc, out, err = run_prog(["-r", "-f", path])
            if split_line(out.strip())[0] == exp_line(v) and rc == 0:
                R.ok(cat)
            else:
                R.fail(cat, "file -f", f"content {content!r}: expected «{exp_line(v)}», got «{out.strip()}» (rc={rc})")
    R.note(cat, f"{len(table)} valid expressions (value and complete line), {len(INVALID_EXPRESSIONS)} invalid, "
                f"{len(NONPOSITIVE_EXPRESSIONS)} ≤ 0, {len(argv_cases)} argv cases, 4 files")


# --------------------------------------------------------------------------------------
# 8. Command line, exit status and readable output

def cat_cli(R):
    cat = "8 cli"

    def expect(desc, args, lines=None, rc=None, out_has=(), out_lines=None, err_has=()):
        r, out, err = run_prog(args, lines)
        problems = []
        if rc is not None and r != rc:
            problems.append(f"rc={r} (expected {rc})")
        for s in out_has:
            if s not in out:
                problems.append(f"stdout without {s!r}")
        for s in err_has:
            if s not in err:
                problems.append(f"stderr without {s!r}")
        if out_lines is not None:
            got = [split_line(l)[0] for l in out.splitlines()]
            if got != out_lines:
                problems.append(f"lines {got} (expected {out_lines})")
        if problems:
            R.fail(cat, "command line", f"{desc}: collatz {' '.join(shlex.quote(a) for a in args)}"
                                        f"{' < ' + repr(lines) if lines is not None else ''}: " + "; ".join(problems)
                                        + f"\n        stdout: {out.strip()[:300]!r}\n        stderr: {err.strip()[:300]!r}")
        else:
            R.ok(cat)

    big = 2 ** 71 + 1
    e27, ebig = exp_line(27), exp_line(big)
    expect("NO → 0", ["-r", "27"], rc=0, out_lines=[e27])
    mid = 2 ** 100 + 1
    expect("NOTMIN with -s → 0", ["-r", "-s", "2^100+1"], rc=0, out_lines=[exp_line(mid, stop=True)])
    expect("NOTMIN without -s → 2", ["-r", "-n", "10", "2^100+1"], rc=2, out_lines=[exp_line(mid, limit=10)])
    expect("-s with x_σ < 2^71 → NO", ["-r", "-s", "2^71+1"], rc=0, out_lines=[exp_line(big, stop=True)])
    expect("error → 1", ["-r", "abc"], rc=1, out_lines=["ERROR invalid_input"], err_has=["collatz: error in"])
    expect("UNKNOWN → 2", ["-r", "-n", "1", "2^71+1"], rc=2, out_lines=[exp_line(big, limit=1)])
    expect("maximum: error + UNKNOWN → 2", ["-r", "-n", "1", "abc", "2^71+1"], rc=2,
           out_lines=["ERROR invalid_input", exp_line(big, limit=1)])
    expect("maximum: NO + error → 1", ["-r", "27", "0"], rc=1, out_lines=[e27, "ERROR not_positive"])
    expect("--max-steps=N", ["-r", "--max-steps=1", "2^71+1"], rc=2, out_lines=[exp_line(big, limit=1)])
    expect("--max-steps N", ["-r", "--max-steps", "1", "2^71+1"], rc=2, out_lines=[exp_line(big, limit=1)])
    expect("-n with '.' separator", ["-r", "-n", "1.000", "2^71+1"], rc=0, out_lines=[exp_line(big, limit=1000)])
    expect("-n with '_' separator", ["-r", "-n", "1_000", "2^71+1"], rc=0, out_lines=[exp_line(big, limit=1000)])
    for bad in ("abc", "-5", "", "1e3", "99999999999999999999999"):
        expect(f"invalid -n {bad!r}", ["-r", "-n", bad, "27"], rc=1, err_has=["invalid value"])
    expect("-n without a value", ["-r", "27", "-n"], rc=1)
    expect("unknown option", ["-r", "-x", "27"], rc=1, err_has=["unknown option"])
    for h in ("-h", "--help"):
        expect(f"help {h}", [h], rc=0, out_has=["Usage: collatz"])
    expect("nonexistent -f", ["-r", "-f", "/nonexistent/collatz-test.txt"], rc=1,
           out_lines=["ERROR file"], err_has=["could not use the file", "file not found"])
    expect("nonexistent -f does not stop the others", ["-r", "27", "-f", "/nonexistent/x.txt", "97"], rc=1,
           out_lines=[e27, "ERROR file", exp_line(97)])
    expect("-f without a name", ["-r", "-f"], rc=1)
    expect("stdin: blanks, comments, CRLF, spaces", ["-r"],
           lines=["", "   ", "# comment", "  27  ", "\t97\r", "#2^71+1", "  # another", "2^71+1"],
           rc=0, out_lines=[e27, exp_line(97), ebig])
    expect("several arguments", ["-r", "27", "97", "2^71+1"], rc=0, out_lines=[e27, exp_line(97), ebig])
    # Readable output
    expect("readable: NO", ["27"], rc=0, out_has=["NOT a counterexample", "σ(n) = 96", "δ(n) = 111"])
    expect("readable: 2^71", ["2^71+1"], rc=0, out_has=["below 2^71", "δ(n) = 565"])
    expect("readable: NOTMIN", ["-s", "2^100+1"], rc=0, out_has=["NOT the SMALLEST counterexample", "does not prove that n converges"])
    expect("readable: -s with x_σ < 2^71", ["-s", "2^71+1"], rc=0, out_has=["falls below 2^71", "σ(n) = 3 steps"])
    expect("readable: plural", ["2"], rc=0, out_has=["σ(n) = 1 step ", "[2 bits · 1 digit]"])
    expect("readable: index without separator", ["2^100-1"], rc=0, out_has=["δ(n) = 1,465 steps  (x_1465 = 1)"])
    expect("readable: -n 0", ["-n", "0", "2^100-1"], rc=2, out_has=["UNKNOWN", "limit of 0 steps"])
    expect("two numbers on one line", ["-r"], lines=["27 31"], rc=1, out_lines=["ERROR invalid_input"],
           err_has=["space between digits"])
    expect("thousands with a space", ["-r", "1 000 027"], rc=0, out_lines=[exp_line(1000027)])
    expect("-n as an expression", ["-r", "-n", "10^3", "2^71+1"], rc=0, out_lines=[exp_line(big, limit=1000)])
    expect("-n 1.5 invalid", ["-r", "-n", "1.5", "27"], rc=1, err_has=["invalid value"])
    expect("readable: UNKNOWN", ["-n", "1", "2^71+1"], rc=2, out_has=["UNKNOWN"])
    expect("readable: n = 1", ["1"], rc=0, out_has=["trivial cycle"])
    expect("readable: -v", ["-v", "27"], rc=0, out_has=["σ in steps of T(x): 59", "δ in steps of T(x): 70"])
    expect("readable: error", ["0"], rc=1, err_has=["not a positive integer"])
    R.note(cat, "exit status, options, standard input and readable output")


# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# 9. Mutation: cases added after testing the suite itself against mutants of the code
#    (each block says which kind of defect it catches; see also the mutation report)

def _parity_word(c, jp):
    """jp T steps, c of them odd, with the odd ones as early as possible (i-th odd one at step
    ceil(i·jp/c)).  With c·log2(3) − jp = −ρ, ρ > 0 tiny, the ratio x_j/x_0 ≈ 3^c_j/2^j stays ≥ 1
    at every step and drops below 1 only at the last one (an even step): σ_T = jp exactly."""
    w, prev = [], 0
    for i in range(1, jp + 1):
        cur = -((-i * c) // jp)
        w.append(1 if cur > prev else 0)
        prev = cur
    return w


def _n_from_parities(w, L, seed=0):
    """An n of L bits whose first len(w) T steps have the parities w (L > len(w))."""
    rng = random.Random(seed)
    X, P3, low, j = 0, 1, 0, 0            # X = T^j(n mod 2^j), P3 = 3^(odd steps so far)
    for s in range(0, len(w), 64):         # 64 steps at a time: T^m(q·2^m + r) = 3^c·q + T^m(r)
        chunk = w[s:s + 64]
        m = len(chunk)
        mask = (1 << m) - 1
        y = ty = cm = 0
        p3s = 1
        for i, p in enumerate(chunk):      # y < 2^m with these parities, bit by bit
            bit = p ^ (ty & 1)
            y |= bit << i
            ty += bit * p3s
            if ty & 1:
                ty = (3 * ty + 1) // 2
                p3s *= 3
                cm += 1
            else:
                ty //= 2
        q = ((y - X) * pow(P3 & mask, -1, 1 << m)) & mask
        low |= q << j
        x = X + P3 * q
        X = p3s * (x >> m) + ty
        P3 *= p3s
        j += m
    top = (1 << (L - 1)) | (rng.getrandbits(max(0, L - len(w) - 1)) << len(w))
    return top | low


def _greedy_word(steps, seed):
    """Parities of `steps` T steps that keep x_j/n in ~[1, 2.3): an even step is taken only if the
    ratio (in log2) stays > 1."""
    rng = random.Random(seed)
    lg3 = 1.5849625007211562
    w, r = [], 0.0
    for _ in range(steps):
        odd = 1 if r < 1.02 + 0.3 * rng.random() else 0
        w.append(odd)
        r += (lg3 - 1) if odd else -1
    return w


def _greedy_hover(L, seed, frac=0.9):
    """n of L bits whose T trajectory stays just above n for ~frac·L steps."""
    return _n_from_parities(_greedy_word(int(L * frac), seed), L, seed)


def _run_after_block(ksafe, run, kind, seed, steps=64):
    """Even n such that n/2 = x1 has ksafe+73 bits (after σ = 1 the safe block has k = ksafe steps) and
    such that, after `steps` T steps, the low bits of x become `run` ones (kind='1') or zeros ('0'):
    a run longer than the k − steps remaining steps (with steps = 64 this happens after the
    1st block of advance_simple; with steps = k/2, after the 1st half of advance_rec)."""
    rng = random.Random(seed)
    L1 = ksafe + 73
    r = rng.getrandbits(steps) | 1 | (1 << (steps - 1))
    ty, cm = r, 0
    for _ in range(steps):
        if ty & 1:
            ty = (3 * ty + 1) // 2
            cm += 1
        else:
            ty //= 2
    target = ((1 << run) - 1) if kind == "1" else 0
    qlow = ((target - ty) * pow(3 ** cm, -1, 1 << run)) % (1 << run)
    qbits = L1 - steps
    q = (1 << (qbits - 1)) | (rng.getrandbits(qbits - run - 1) << run) | qlow
    return 2 * ((q << steps) | r)


def cat_mutation(R):
    cat = "9 mutation"
    tmp = tempfile.mkdtemp(prefix="collatz-mut-")

    def ok(cond, kind, detail):
        if cond:
            R.ok(cat)
        else:
            R.fail(cat, kind, detail() if callable(detail) else detail)

    def runb(args, data=None, merge=False, timeout=120):
        p = subprocess.run([CFG.bin] + list(args), input=data, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT if merge else subprocess.PIPE, timeout=timeout)
        return p.returncode, p.stdout.decode("utf-8", "replace"), (b"" if merge else p.stderr)

    def raw_head(line):
        return split_line(line)[0]

    # ---- 9.1 Near misses of the watch: the first crossing x_k < n falls on the last possible step ----
    # ρ = jp − c·log2(3) > 0 tiny: x_{jp−1}/n ≈ 2·(1 − ρ·ln2) and the next even step crosses n by
    # a relative margin of ~ρ. An error of ε per odd step in the log2 bound would make the watch skip
    # the crossing (σ_T = jp + 1 or more) when ε·(odd steps of the last block of 64) > ρ.
    near = [(5, 8), (41, 65), (306, 485), (14936, 23673), (15601, 24727), (190537, 301994)]
    for c, jp in near:
        n0 = _n_from_parities(_parity_word(c, jp), jp + 100, seed=c)
        low = n0 & ((1 << jp) - 1)
        nvar = 16 if jp > 100000 else 2      # different high bits change the alignment of the watched blocks
        ns = [n0]
        for sd in range(1, nvar):
            L = jp + 100 + sd % 3
            ns.append((1 << (L - 1)) | (random.Random(sd).getrandbits(L - jp - 1) << jp) | low)
        for vi, n in enumerate(ns):
            text = hex(n)
            want = f"NOTMIN sigma={jp + c} sigmaT={jp} bound=- total=- totalT=- bits={n.bit_length()}"
            if jp <= 30000 and vi == 0:     # the reference confirms the design (σ_T = jp) where it is cheap
                ev = ref.events(n, max_steps=jp + c + 10, stop_at_sigma=True)
                ok(ref.format_line(ref.expected_line(n, stop=True, ev=ev)) == want, "near-miss design",
                   f"c={c} jp={jp}: reference {ev['sigma']}/{ev['sigmaT']} ≠ design {jp + c}/{jp}")
            variants = [[], ["--simple"]] + ([["--no-blocks"]] if vi == 0 else []) \
                + ([["--no-native"]] if jp < 1000 else [])
            results = run_many([(["-r", "-s"] + v, [text]) for v in variants])
            for v, (rc, out, err) in zip(variants, results):
                line = out.strip()
                ok(rc == 0 and raw_head(line) == want, "near miss: crossing skipped by the watch",
                   lambda: f"n with {n.bit_length()} bits (c={c}, jp={jp}, variant {vi}) [-s {' '.join(v)}]\n"
                           f"        expected: {want}\n        got:      {line}\n"
                           f"        repro:    {repro(['-s'] + v, text)}")
            if vi > 0:
                continue
            # step limits around σ (rem/2 of the watched block) and in the middle
            for N in (jp + c - 1, jp + c, jp + c + 1, (jp + c) // 2, 7):
                exp = want if N >= jp + c else f"UNKNOWN sigma=- sigmaT=- bound=- total=- totalT=- bits={n.bit_length()}"
                for v in ([], ["--simple"]):
                    rc, out, err = run_prog(["-r", "-s", "-n", str(N)] + v, [text])
                    ok(raw_head(out.strip()) == exp and rc == (0 if N >= jp + c else 2), "near miss with -n",
                       f"(c={c}, jp={jp}) -s -n {N} {' '.join(v)}: expected {exp}, got {out.strip()!r} rc={rc}")

    # ---- 9.2 Trajectories that stay just above n (the adversarial class for the watch) ----
    for L, seed in ((400, 1), (3000, 2), (12000, 3)):
        n = _greedy_hover(L, seed)
        text = hex(n)
        evs = ref.events(n, stop_at_sigma=True)
        sg = evs["sigma"]
        for N in (None, sg - 1, sg, sg + 1, sg // 3):
            ev_n = evs if N is None else ref.events(n, max_steps=N, stop_at_sigma=True)
            exp = ref.format_line(ref.expected_line(n, stop=True, limit=N, ev=ev_n))
            flags = ["-s"] + ([] if N is None else ["-n", str(N)])
            variants = [[], ["--simple"], ["--no-blocks"]] + ([["--no-native"]] if L < 1000 else [])
            for v, (rc, out, err) in zip(variants, run_many([(["-r"] + flags + v, [text]) for v in variants])):
                ok(raw_head(out.strip()) == exp, "trajectory hugging n",
                   lambda: f"hover L={L} seed={seed} [{' '.join(flags + v)}]\n        expected: {exp}\n"
                           f"        got:      {out.strip()}\n        repro:    {repro(flags + v, text)}")
        if L == 400:     # full mode (down to 1) too
            exp = exp_line(n, False, None, ref.events(n))
            for v in VARIANTS:
                rc, out, err = run_prog(["-r"] + v, [text])
                ok(raw_head(out.strip()) == exp, "trajectory hugging n (full)",
                   f"hover L=400 [{' '.join(v)}]: expected {exp}, got {out.strip()}")

    # the crossing (even step with ratio < 1) falls exactly at the start of a long watched block: the child
    # of the advance_rec split takes no step at all (jj = 0) and x must be rebuilt with the full
    # shift; (K, L) were found by sweeping K with an instrumented binary.
    w11 = _greedy_word(27580, 11)
    for K, L in ((507, 1214), (4035, 4165), (27579, 27709)):
        n = _n_from_parities(w11[:K] + [0], L, seed=K)
        ev = ref.events(n, stop_at_sigma=True)
        for stop_flags in (["-s"], ["-s", "-n", str(ev["sigma"] - 1)], ["-s", "-n", str(ev["sigma"])]):
            lim = None if "-n" not in stop_flags else int(stop_flags[-1])
            exp = ref.format_line(ref.expected_line(n, stop=True, limit=lim, ev=ev))
            for v, (rc, out, err) in zip([[], ["--simple"]], run_many([(["-r"] + stop_flags + v, [hex(n)])
                                                                        for v in ([], ["--simple"])])):
                ok(raw_head(out.strip()) == exp, "crossing at the start of a watched block",
                   lambda: f"K={K} L={L} {' '.join(stop_flags + v)}: expected {exp}, got {out.strip()}")

    # ---- 9.3 Performance: an n of 302 thousand bits hugging n must not cost O(L²) ----
    # The computation time ("time=" of the program itself, the smallest of 3 runs) is ~0.002 s with the watch
    # and ~0.4 s without it (exact runs only, one pass per limb at each step); the 0.15 s limit leaves a
    # 75x margin for slow or busy machines.
    c, jp = near[-1]
    text = hex(_n_from_parities(_parity_word(c, jp), jp + 100, seed=c))
    best = None
    for _ in range(3):
        rc, out, err = run_prog(["-r", "-s"], [text])
        m = re.search(r" time=(\d+\.\d+)", out)
        if rc == 0 and m:
            best = float(m.group(1)) if best is None else min(best, float(m.group(1)))
    ok(best is not None and best < 0.15, "performance regression (watch)",
       f"302-thousand-bit near miss with -s: best computation time {best} s (expected ~0.002 s; limit 0.15 s)")

    # ---- 9.4 Runs of 0s/1s that appear AFTER the 1st block of 64 steps of a short block ----
    cases, evs_list = [], []
    for ksafe, run, steps in ((130, 76, 64), (192, 130, 64), (256, 200, 64), (256, 130, 64), (200, 140, 64),
                              (200, 110, 100), (240, 130, 120), (256, 138, 128), (280, 150, 140),
                              (400, 210, 200), (1000, 520, 500)):
        for kind in "01":
            n = _run_after_block(ksafe, run, kind, seed=ksafe * 7 + run, steps=steps)
            cases.append(mk(n, f"run of {run} '{kind}' after {steps} steps of a block of k={ksafe}"))
    evs_list = ref_events([c.n for c in cases])
    check_cases(R, cat, cases, evs_list, VARIANTS)

    # ---- 9.4b 3^e with large e (pow3 by squaring, e ≥ 50000, even and odd): 2^k−1 has k trailing ones ----
    # (in 5 consistency, these sizes are only compared between variants, which use the same pow3)
    ks = (50000, 50001, 65537)
    big_cases_k = [mk((1 << k) - 1, f"2^{k}-1") for k in ks]
    evs_k = ref_events([c.n for c in big_cases_k], stop_at_sigma=True)
    check_cases(R, cat, big_cases_k, evs_k, [[], ["--no-threads"], ["--simple"]], modes=(True,))

    # ---- 9.5 Readable output: values and labels ----
    def hr(args, lines=None, merge=False):
        data = None if lines is None else "".join(s + "\n" for s in lines).encode()
        return runb(args, data, merge)

    # x_σ with up to 130 bits is printed in full; from 131 to 200 bits, abbreviated (12 + 12 digits)
    for bits in (110, 150, 199):
        n = (1 << (bits - 1)) + 1
        x, k = n, 0
        while x >= n:
            x = x // 2 if x % 2 == 0 else 3 * x + 1
            k += 1
        for v in ([], ["--no-native"], ["--no-blocks"]):
            rc, out, err = hr(["-s"] + v + [str(n)])
            xs = str(x) if x.bit_length() <= 130 else f"{str(x)[:12]}…{str(x)[-12:]} ({len(str(x))} digits)"
            ok(f"(x_{k} = {xs} < n)" in out, "value of x_σ in the readable output",
               f"n=2^{bits - 1}+1 {' '.join(v)}: missing «(x_{k} = {xs} < n)» in {out!r}")
    n = (1 << 260) + 1                                # 261 bits: x_σ has >200 bits → «a number with N bits»
    rc, out, err = hr(["-s", str(n)])
    ok("a number with" in out and "bits < n)" in out, "large x_σ is abbreviated", out)

    # size: exact up to 20000 bits, approximate (≈) beyond; plural
    rc, out, err = hr(["-s", "2^5000"])
    ok(f"[5,001 bits · {len(str(2 ** 5000)):,} digits]" in out, "exact size (≤ 20000 bits)", out)
    rc, out, err = hr(["-s", "2^30000"])
    ok("≈" in out, "approximate size (> 20000 bits)", out)

    # long label: 30 characters, «…», 30 characters; without cutting UTF-8 in half
    label = "+".join(["1"] * 41)
    rc, out, err = hr([label])
    ok("n = " + label[:30] + "…" + label[-30:] in out, "long label abbreviated", out)
    rc, out, err = hr(["x" + "é" * 50 + "x"])
    try:
        err.decode("utf-8")
        valid = rc == 1
    except UnicodeDecodeError:
        valid = False
    ok(valid, "long label cut in the middle of a UTF-8 character", repr(err))
    rc, out, err = hr(["-r"], ["2\x07x"])
    ok(b"\x07" not in err and b"2?x" in err, "control character in the label", repr(err))
    rc, out, err = hr(["2^100"])
    ok("= 1267650600228229401496703205376" in out, "expression value in the readable output", out)
    rc, out, err = hr(["-n", "10", "2^100+1"])
    ok("was reached before" in out and "δ(n)" in out, "what the limit prevented from being computed", out)
    rc, out, err = hr(["-s", "2^100+1"])
    ok("run without -s" in out, "complete-proof hint with -s", out)
    ok(re.search(r"computation time: \d+\.\d+ ", out) is not None, "time with a decimal point", out)
    rc, out, err = hr(["-r", "27"])
    ok(re.search(r" time=\d+\.\d{6}\n$", out) is not None, "raw time with 6 decimals", out)
    path = os.path.join(tmp, "big.txt")
    with open(path, "w") as f:
        f.write("1" * 50 + "\n")
    rc, out, err = hr(["-s", "-f", path])
    ok("n = 111111111111…111111111111 (file " in out, "number from a file abbreviated", out)

    # messages per verdict (readable output)
    rc, out, err = hr(["-s", "1000"])
    ok("n < 2^71" in out and "below 2^71  " not in out and "falls below 2^71" not in out, "-s with n < 2^71", out)
    rc, out, err = hr(["27"])
    ok("the trajectory reaches 1" in out and "below 2^71  " not in out, "n < 2^71 reaches 1", out)
    rc, out, err = hr(["1"])
    ok("trivial cycle" in out and "σ(1) does not apply" in out and "δ(1) = 0 steps" in out, "n = 1", out)
    rc, out, err = hr(["-n", "5", "2^100+1"])
    ok("increase the limit -n" in out and rc == 2, "NOTMIN by limit without -s", out)
    rc, out, err = hr(["-n", "1", "2^71+1"])
    ok("UNKNOWN" in out and "Increase the limit -n" in out and rc == 2, "UNKNOWN", out)
    rc, out, err = hr(["27", "97"])
    ok("\n\nn = 97" in out, "blank line between results", out)

    # ---- 9.6 Options ----
    big = 2 ** 71 + 1
    mid = 2 ** 100 + 1

    def cli(desc, args, lines=None, rc=None, out_lines=None, out_has=(), err_has=()):
        r, out, err = hr(args, lines)
        got = [raw_head(l) for l in out.splitlines()] if out_lines is not None else None
        probs = []
        if rc is not None and r != rc:
            probs.append(f"rc={r} (expected {rc})")
        if out_lines is not None and got != out_lines:
            probs.append(f"lines {got} (expected {out_lines})")
        probs += [f"stdout without {s!r}" for s in out_has if s not in out]
        probs += [f"stderr without {s!r}" for s in err_has if s.encode() not in err]
        ok(not probs, "option/output", f"{desc}: collatz {' '.join(shlex.quote(a) for a in args)}: {'; '.join(probs)}"
                                       f"\n        stdout: {out.strip()[:200]!r}\n        stderr: {err.decode(errors='replace').strip()[:200]!r}")

    lim5 = exp_line(mid, limit=5)
    for form in (["-n", "5"], ["-n5"], ["-n=5"], ["--max-steps=5"], ["--max-steps", "5"]):
        cli(f"limit {form}", ["-r"] + form + [str(mid)], rc=2, out_lines=[lim5])
    cli("-n = 2^64−1", ["-r", "-n", "18446744073709551615", "27"], rc=0, out_lines=[exp_line(27)])
    cli("-n = 2^64", ["-r", "-n", "18446744073709551616", "27"], rc=1, err_has=["invalid value"])
    cli("-n with a space between digits", ["-r", "-n", "27 31", "27"], rc=1, err_has=["invalid value"])
    for bad in ("1234.567", ".000", "1.000+5", "1.00", "1.0000", "1..000", "1.000.00"):
        cli(f"-n {bad!r} invalid", ["-r", "-n", bad, "27"], rc=1, err_has=["invalid value"])
    cli("-n thousands dots with more than 63 characters", ["-r", "-n", "1" + ".000" * 30, "27"], rc=1,
        err_has=["invalid value"])
    cli("-n 12.345.678", ["-r", "-n", "12.345.678", "27"], rc=0, out_lines=[exp_line(27)])
    cli("-rs combined", ["-rs", str(mid)], rc=0, out_lines=[exp_line(mid, stop=True)])
    cli("-sr combined", ["-sr", str(mid)], rc=0, out_lines=[exp_line(mid, stop=True)])
    cli("-sv combined", ["-sv", "27"], rc=0, out_has=["σ in steps of T(x): 59"])
    cli("rc = maximum (UNKNOWN before the error)", ["-r", "-n", "1", str(big), "abc"], rc=2,
        out_lines=[exp_line(big, limit=1), "ERROR invalid_input"])
    cli("rc = maximum on standard input", ["-r", "-n", "1"], lines=[str(big), "abc"], rc=2)
    cli("rc = maximum with nonexistent -f", ["-r", "-n", "1", str(big), "-f", "/nonexistent/x.txt"], rc=2,
        out_lines=[exp_line(big, limit=1), "ERROR file"])
    cli("empty standard input", ["-r"], lines=[], rc=1, err_has=["no number"])
    cli("standard input with comments only", ["-r"], lines=["", "# nothing"], rc=1, err_has=["no number"])
    cli("standard input: BOM", ["-r"], lines=["﻿27", "97"], rc=0, out_lines=[exp_line(27), exp_line(97)])
    cli("standard input: # in the middle of the line", ["-r"], lines=["27 # note", "97#x"], rc=0,
        out_lines=[exp_line(27), exp_line(97)])

    # order between stdout and stderr (2>&1): the program flushes stdout before each error message
    # (the 1st line is flushed at once; the following ones stay in the buffer for up to 0.2 s, so the error must flush them)
    rc, out, _ = runb(["-r", "27", "97", "abc", "27", "97", "0"], merge=True)
    kinds = ["E" if l.startswith("collatz:") else "S" for l in out.splitlines()]
    ok(kinds == ["S", "S", "E", "S", "S", "S", "E", "S"], "stdout/stderr order (2>&1)", out)

    # raw output flushed promptly (at most 0.2 s later), even with the input still open
    import select
    p = subprocess.Popen([CFG.bin, "-r"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    got_lines = []
    try:
        for text in ("27", "97"):
            p.stdin.write((text + "\n").encode())
            p.stdin.flush()
            t0 = time.time()
            buf = b""
            while b"\n" not in buf and time.time() - t0 < 5:
                if select.select([p.stdout], [], [], 0.1)[0]:
                    buf += os.read(p.stdout.fileno(), 4096)
            got_lines.append(buf.decode().strip())
            time.sleep(0.3)
    finally:
        p.stdin.close()
        p.wait(timeout=10)
    ok([raw_head(l) for l in got_lines] == [exp_line(27), exp_line(97)], "raw output is not flushed in time",
       f"lines received before closing the input: {got_lines!r}")

    # progress on the terminal (stderr is a TTY): lines with \r during the computation and, at the end, the line is cleared.
    # It appears only after 1 s of computation: the size grows (600 thousand, 1 million, 1.6 million bits) until the
    # run lasts ≥ 1.3 s; on a machine that is too fast the test is skipped (with a note).
    import pty
    done_tty = False
    for bits in (600000, 1000000, 1600000):
        path = os.path.join(tmp, "big-slow.txt")
        rng = random.Random(5)
        with open(path, "w") as f:
            f.write(hex(rng.getrandbits(bits) | (1 << (bits - 1))))
        master, slave = pty.openpty()
        t0 = time.time()
        p = subprocess.Popen([CFG.bin, "-r", "--no-blocks", "-f", path], stdout=subprocess.PIPE, stderr=slave)
        os.close(slave)
        tty = b""
        while True:
            if select.select([master], [], [], 0.2)[0]:
                try:
                    chunk = os.read(master, 4096)
                except OSError:          # the other side closed (end of the program)
                    break
                if not chunk:
                    break
                tty += chunk
            elif p.poll() is not None:
                break
        out_tty = p.stdout.read().decode()
        p.wait()
        os.close(master)
        if time.time() - t0 >= 1.3:
            ok(b"steps; current x has" in tty and tty.endswith(b"\r" + b" " * 78 + b"\r") and out_tty.startswith("NO"),
               "progress on the terminal", f"stderr (tty) ended with {tty[-120:]!r}; stdout {out_tty[:60]!r}")
            done_tty = True
            break
    if not done_tty:
        R.note(cat, "progress on the terminal not tested: the machine solved 1.6 million bits in < 1.3 s")

    # ---- 9.7 Files (-f) ----
    def fcase(desc, content, rc=None, out_lines=None, err_has=(), err_not=()):
        path = os.path.join(tmp, "f%d.txt" % abs(hash(desc)))
        with open(path, "wb") as f:
            f.write(content)
        r, out, err = hr(["-r", "-f", path])
        got = [raw_head(l) for l in out.splitlines()]
        probs = []
        if rc is not None and r != rc:
            probs.append(f"rc={r} (expected {rc})")
        if out_lines is not None and got != out_lines:
            probs.append(f"lines {got} (expected {out_lines})")
        probs += [f"stderr without {s!r}" for s in err_has if s.encode() not in err]
        probs += [f"stderr with {s!r}" for s in err_not if s.encode() in err]
        ok(not probs, "file -f", f"{desc}: {content[:60]!r}: {'; '.join(probs)}; stderr {err.decode(errors='replace')[:200]!r}")

    v100 = exp_line(2 ** 100 - 1)
    fcase("NUL", b"27\x00", rc=1, out_lines=["ERROR file"], err_has=["null bytes"])
    fcase("UTF-16", "27\n".encode("utf-16"), rc=1, out_lines=["ERROR file"], err_has=["null bytes"])
    fcase("BOM", b"\xef\xbb\xbf27\n", rc=0, out_lines=[exp_line(27)])
    fcase("comments", b"# header\n2^100 # middle\n - 1 # end\n", rc=0, out_lines=[v100])
    fcase("comment only", b"# nothing\n", rc=1, out_lines=["ERROR invalid_input"])
    fcase("number split across lines without groups of 3", b"12345\n67890\n", rc=0, out_lines=[exp_line(1234567890)],
          err_has=["warning"])
    fcase("loose digit groups in the file", b"27 31\n", rc=0, out_lines=[exp_line(2731)])
    d30, e30, f30 = "1" * 30, "2" * 30, "3" * 30
    for desc, content, warn in (
            ("2 equal lines of 30", f"{d30}\n{e30}\n", False),
            ("3 lines, last one shorter", f"{d30}\n{e30}\n{'4' * 5}\n", False),
            ("1 line", f"{d30}\n", False),
            ("short lines (<20)", "123\n456\n", True),
            ("middle line of another length", f"{d30}\n{'2' * 25}\n{f30}\n", True),
            ("last line longer than the 1st", f"{d30}\n{'2' * 35}\n", True)):
        ok_n = int(content.replace("\n", ""))
        fcase("warning: " + desc, content.encode(), rc=0, out_lines=[exp_line(ok_n)],
              err_has=[] if not warn else ["warning"], err_not=["warning"] if not warn else [])

    # ---- 9.8 Expressions: limits and messages ----
    def expr(text, want, **kw):
        cli(f"expression {text[:30]!r}", ["-r", "--", text], **{"rc": 0 if want else 1, **kw},
            out_lines=[want] if want else ["ERROR invalid_input"])

    expr("(" * 2000 + "7" + ")" * 2000, exp_line(7))
    expr("(" * 2001 + "7" + ")" * 2001, None)
    expr("1" + "^1" * 2000, exp_line(1))
    expr("1" + "^1" * 2001, None)
    expr("(1)+" * 2500 + "(1)", exp_line(2501))               # the depth goes back down on closing ')'
    expr("1^1+" * 2500 + "1", exp_line(2501))                 # ... and on finishing an exponent
    expr("9" * 39, exp_line(10 ** 39 - 1))                    # 39 digits > 2^128: does not fit the 38-digit shortcut
    expr("9" * 38, exp_line(10 ** 38 - 1))
    expr("340282366920938463463374607431768211455", exp_line(2 ** 128 - 1))
    expr("340282366920938463463374607431768211456", exp_line(2 ** 128))
    for text, err_has in (("2* *x", ["number expected (position 4)"]),
                          ("2+", ["(position 3), but the input ended"]),
                          ("(1", ["')' expected (position 3), but the input ended"]),
                          ("1 + x", ["(position 5): 'x'"]),
                          ("é1 22", ["space between digits (position 4)"]),
                          ("3×4", ["(position 2)", "U+00D7", "use * to multiply"]),
                          ("1,5", ["separate thousands"]),
                          ("1e5", ["scientific notation"]),
                          ("2^-1", ["negative exponent (position 2)"]),
                          ("7/0", ["division by zero (position 2)"])):
        cli(f"error message {text!r}", ["-r", "--", text], rc=1, err_has=err_has)


CATEGORIES = [("exhaustive", cat_exhaustive), ("known", cat_known), ("random", cat_random),
              ("structured", cat_structured), ("consistency", cat_consistency), ("limits", cat_limits),
              ("expressions", cat_expressions), ("cli", cat_cli), ("mutation", cat_mutation)]


def main():
    ap = argparse.ArgumentParser(description="Differential tests of the collatz verifier.")
    ap.add_argument("--long", action="store_true", help="more cases and larger numbers (several minutes)")
    ap.add_argument("--only", default="", help="comma-separated categories (name or number 1-9): "
                                               + ",".join(name for name, _ in CATEGORIES))
    ap.add_argument("--bin", default=os.environ.get("COLLATZ_BIN"), help="binary to test (default: ./collatz)")
    ap.add_argument("--jobs", type=int, default=CFG.jobs, help="parallel processes")
    ap.add_argument("--seed", type=int, default=CFG.seed, help="seed of the random tests")
    ap.add_argument("--examples", type=int, default=CFG.max_examples, help="examples shown per kind of failure")
    ap.add_argument("--timeout", type=float, default=None,
                    help=f"seconds per program run (default {CFG.timeout}, 1800 with --long)")
    ap.add_argument("--stop", action="store_true", help="stop at the first category with a failure")
    args = ap.parse_args()

    CFG.long, CFG.jobs, CFG.seed, CFG.max_examples = args.long, max(1, args.jobs), args.seed, args.examples
    CFG.timeout = args.timeout if args.timeout is not None else (1800 if args.long else CFG.timeout)
    if args.bin:
        CFG.bin = os.path.abspath(args.bin)
        CFG.bin_display = args.bin
    if not os.access(CFG.bin, os.X_OK):
        print(f"binary not found: {CFG.bin} (run `make` first)", file=sys.stderr)
        return 1
    src = os.path.join(ROOT, "collatz.c")
    if not args.bin and os.path.exists(src) and os.path.getmtime(src) > os.path.getmtime(CFG.bin):
        print("warning: collatz.c is newer than the ./collatz binary (run `make`)", file=sys.stderr)

    wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    known = {name for name, _ in CATEGORIES} | {str(i) for i in range(1, len(CATEGORIES) + 1)}
    if wanted - known:
        print(f"unknown category: {', '.join(sorted(wanted - known))} "
              f"(use {', '.join(name for name, _ in CATEGORIES)} or 1-{len(CATEGORIES)})", file=sys.stderr)
        return 1
    R = Results()
    t_all = time.time()
    print(f"Testing {CFG.bin} ({'--long' if CFG.long else 'default'}, {CFG.jobs} processes, seed {CFG.seed})")
    for i, (name, fn) in enumerate(CATEGORIES, 1):
        if wanted and name not in wanted and str(i) not in wanted:
            continue
        t0 = time.time()
        print(f"  [{i}] {name} ...", end="", flush=True)
        before = sum(c["fail"] for c in R.cats.values())
        fn(R)
        catname = next(k for k in R.cats if k.startswith(str(i)))
        R.cats[catname]["time"] = time.time() - t0
        fails = sum(c["fail"] for c in R.cats.values()) - before
        print(f" {time.time() - t0:.1f}s, {R.cats[catname]['ok'] + R.cats[catname]['fail']} checks"
              + (f", {fails} FAILURES" if fails else ""))
        if fails and args.stop:
            break
    if _POOL is not None:
        _POOL.shutdown()
    R.summary()
    print(f"total time: {time.time() - t_all:.1f}s")
    if CFG.faildir:
        print(f"long inputs of the failures saved in {CFG.faildir}")
    return 1 if R.any_failure() else 0


if __name__ == "__main__":
    sys.exit(main())
