# Collatz counterexample verifier

`collatz` decides whether a positive integer `n` (up to millions of digits) is a counterexample to
the Collatz conjecture (x → x/2 if even, x → 3x+1 if odd), i.e. whether its trajectory diverges or
enters a cycle other than 1 → 4 → 2 → 1. `candidates` lists the `n` that may still be the
**smallest** counterexample and discards the rest by sieves (see
[Candidate generator](#candidate-generator)).

## Build

```sh
brew install gmp      # big-integer library
make                  # builds ./collatz and ./candidates
make test             # tests of both programs (~2 min)
make bench            # performance table (bench/bench.py)
```

## Usage

```sh
./collatz "2^100000-1"                 # expressions: + - * / % ^ ( ), 0x… and 0b…
./collatz -s "2^1000000-1"             # stopping time only (fastest criterion)
./collatz -f huge_number.txt           # ONE number from a file (line breaks ignored)
seq -f %.0f 1 1000000 | ./collatz -r   # one number per line (%.0f avoids "1e+06" on macOS)
```

```
$ ./collatz "2^100-1"
n = 2^100-1 = 1267650600228229401496703205375   [100 bits · 31 digits]
✔ NOT a counterexample: the trajectory reaches 1.
  • stopping time          σ(n) = 592 steps  (x_592 = 1229924552300801959567057140184 < n)
  • below 2^71             after 862 steps  (every m < 2^71 has already been verified)
  • total stopping time    δ(n) = 1,465 steps  (x_1465 = 1)
  computation time: 6.7 µs
```

| option | effect |
|---|---|
| `-s`, `--stopping` | stop at the stopping time σ(n), the first k with x_k < n |
| `-f`, `--file FILE` | read one number from a file |
| `-n`, `--max-steps N` | give up after N steps (`1000000`, `1.000.000`, `10^6`) |
| `-r`, `--raw` | compact output: `VERDICT sigma= sigmaT= bound= total= totalT= bits= time=` |
| `-v`, `--verbose` | steps of the accelerated map T and parsing time |
| `--data` | also write trajectory data to `Tested/data.csv` |
| `--no-record` | do not record verified numbers in `Tested/` |
| `-h`, `--help` | help |

Combined and attached forms are accepted (`-sv`, `-n5`, `--max-steps=5`, `--file=FILE`); `--` ends
the options. Input digits may be grouped by `_`, dots or spaces in groups of 3.

| verdict (`-r`) | meaning | exit code |
|---|---|---|
| `NO` | proved not a counterexample | `0` |
| `NOTMIN` | proved only "not the smallest counterexample" (sufficient with `-s`) | `0` |
| `ERROR` | input error, followed by the reason (also: no number on standard input) | `1` |
| `UNKNOWN` | the `-n` limit ran out before the requested conclusion | `2` |
| `CYCLE` | counterexample found (non-trivial cycle) | `3` |

With several numbers, the highest code wins. In raw fields, `-` means "not reached", `bound=0`
means n < 2^71, `sigmaT`/`totalT` count steps of T, and `time` is in seconds.

## Interruption

Ctrl+C, `kill` or closing the terminal stops the program at a safe point within a fraction of a
second: input reading stops, the unproved number in progress is discarded, output ends at the last
complete line, proved numbers are recorded and `Tested/` is consolidated (exit by the signal,
shell code 130). During a huge number the exit is immediate and the next run consolidates. A 2nd
Ctrl+C exits at once, keeping what is proved; a 3rd exits unconditionally. A closed output
(`| head`) stops `collatz` the same way (code 141). Only `kill -9` or power loss can lose the last
second of work; the record stays consistent in every case.

## Record of tested numbers

Every n in [2^71, 2^127) with verdict `NO` is stored in `Tested/` and is not generated again by
`candidates`. `NOTMIN`, `UNKNOWN`, `CYCLE` and errors are never recorded. The folder lies next to
the executable; `COLLATZ_TESTED` overrides it.

```sh
./candidates '2^71' '2^71+10^9' | ./collatz -r -s   # verify and record
./candidates '2^71' '2^71+10^9' | ./collatz -r -s   # again: nothing to do
./candidates --tested | head                        # tested numbers, ascending (-c: count)
```

| file | content |
|---|---|
| `tested.bin` | all tested numbers, ascending, without duplicates, compressed |
| `tested.journal.<pid>` | journal of each running (or killed) `collatz` |
| `tested.journal` | finished numbers not yet merged into `tested.bin` |
| `data.csv` | with `--data`: trajectory data per tested number, ordered by n |
| `data.journal.<pid>.csv`, `data.journal.csv` | data lines not yet merged |
| `.lock` | coordinates rewriting of `tested.bin` and `data.csv` |

* **Format.** First number of each 4096-number block, then differences (~2 bytes each instead of
  23 as text: 2,538,564 numbers take 4.5 MB instead of 58.4 MB). A block index allows reading
  only a range. Header, index and blocks carry CRC-32.
* **Writing.** Every ~1 s to the process's own journal, without locking. At exit, journals are
  merged into `tested.bin` through a temporary file and an atomic rename; journals of killed
  processes are merged by the next run. Above ~4 million entries, `tested.bin` is rewritten only
  when the journals exceed 1/16 of it.
* **Integrity.** Concurrent processes are safe. Truncated records, wrong CRCs and corrupted blocks
  are ignored with a warning (those numbers are simply regenerated); a corrupted `tested.bin` is
  never rewritten. A legacy `tested.txt` is imported once and renamed to `tested.txt.imported`.

### Trajectory data (`--data`)

One line per stored number, for the map x → x/2, x → 3x+1:

```
n,k,odd_k,j_minus_k,odd_j_minus_k,exponent_j,m,x_m,odd_m
2361183241434822606851,6,2,555,189,4,3,10625324586456701730832,2
```

| column | meaning |
|---|---|
| `k` | stopping time σ: first k ≥ 1 with x_k < x_0 |
| `odd_k` | odd terms among x_0, …, x_{k−1} |
| `j_minus_k` | j − k, where x_j = 2^e is the first power of 2 of the trajectory (negative if reached before x_k) |
| `odd_j_minus_k` | odd terms among x_k, …, x_{j−1} |
| `exponent_j` | e in x_j = 2^e (total stopping time = j + e) |
| `m`, `x_m` | index (first occurrence) and value of the trajectory maximum |
| `odd_m` | odd terms among x_0, …, x_{m−1} |

The data follow the full trajectory to 1, even with `-s`, and are checked against σ and δ (cost
~1 µs and ~70 bytes per number).

For analysis, convert to SQLite with `make sqlite` (`tools/data_to_sqlite.py [DIR] [-o OUT]`, no
dependencies; ~3 s per million rows). The script rebuilds `Tested/data.sqlite` from `data.csv` and
the data journals, with table `data`: the CSV columns plus `bits`, `x_m_bits` and `total` (δ = k +
j_minus_k + exponent_j). `n` and `x_m` are stored as exact decimal TEXT; rows are in ascending
order of n (`ORDER BY rowid`).

```sh
sqlite3 -header -column Tested/data.sqlite "SELECT n, k, total FROM data ORDER BY total DESC LIMIT 10"
```

## Criteria

1. **Stopping time σ(n)**, the first k ≥ 1 with x_k < x₀. Fastest; alone it proves only that n is
   **not the smallest** counterexample (the minimal element of a non-trivial cycle has no σ, and
   the cycle is detected). In an **ascending search** this suffices.
2. **Below 2^71.** Every integer below 2^71 ≈ 2.36·10²¹ is verified (D. Barina, *The Journal of
   Supercomputing* 81, 2025), so x_k < 2^71 proves n.
3. **Total stopping time δ(n)**, the first k with x_k = 1: a self-contained proof.

Without `-s`, the run goes to 1 at almost the cost of reaching 2^71. With `-s`, it stops at σ(n):
the verdict is `NO` if n < 2^71 or x_σ < 2^71, otherwise `NOTMIN`. Non-trivial cycles are detected
by Brent's algorithm; divergence cannot be proved by computation and runs until the `-n` limit.

## Implementation

* **Accelerated map** T(x) = x/2 or (3x+1)/2; counts are converted to Collatz steps.
* **Native arithmetic up to 128 bits:** 8 steps per table lookup away from thresholds; near them,
  exact runs (x+1 = 2^a·m, m odd ⇒ x ← 3^a·m − 1, then x ← x/2^ctz(x)); below 2^12, a table.
* **2-adic identity** (GMP): T^k(q·2^k + r) = 3^c·q + T^k(r), c = odd steps of r. Applied
  recursively, k steps cost O(M(k)·log k) instead of O(k²). Large products are split across
  performance cores.
* **Safe blocks:** since T(x) ≥ x/2, with x of L bits and threshold t (n, 2^71 or 2) of B bits, the
  next L−1−B steps cannot cross t.
* **Guarded blocks:** log₂ x_j ≥ log₂ x + c·log₂ 3 − j, tracked in fixed point and rounded down,
  bounds how far to jump without skipping the first x_k < t.

## Performance

Apple M5, computation time only, best of 5 on an idle machine:

| input | to 1 (default) | σ only (`-s`) |
|---|---|---|
| 64-bit number | < 1 µs | < 1 µs |
| 2^1000 − 1 | 4 µs | 2 µs |
| 2^10000 − 1 | 75 µs | 38 µs |
| 2^100000 − 1 | 1.9 ms | 1.0 ms |
| 2^1000000 − 1 (301 thousand digits) | 33 ms | 15 ms |
| 2^10000000 − 1 (3 million digits) | 0.52 s | 0.23 s |
| random, 10⁷ bits | 0.29 s | 85 µs |
| constructed worst case (10⁶ bits, near n for 1.6 million steps) | 25 ms | 7 ms |

With `-s`, parsing a huge decimal number costs more than the computation; expressions (`2^N-1`)
and hexadecimal input parse much faster.

## Tests

`tests/reference.py` is a deliberately simple step-by-step simulation, checked against published
records of δ(n), σ(n) and excursion (Oliveira e Silva; Barina). `tests/test_collatz.py` compares
the program with it on every n up to 200,000 (2 million with `--long`), random numbers, 2^k±1,
3^k±1, neighbours of 2^64, 2^71 and 2^128, the `-n` limit, expressions and the command line, and
checks that all internal paths agree on numbers up to 10⁶ bits. The suite passes under
AddressSanitizer and UndefinedBehaviorSanitizer.

# Candidate generator

If a counterexample exists, a smallest one N ≥ 2^71 exists. `candidates` lists the n in [A, B) that
may still be N and discards all others by sieves, **without following any trajectory**. Guarantee:
if N lies in [A, B), it is listed. Numbers already in `Tested/` are neither generated nor counted.

```sh
./candidates -q 5                                 # first 5 candidates from 2^71
./candidates '2^71' '2^71+10^9' | ./collatz -r -s # generate and verify (~1 s)
./candidates -c '2^71' '2^71+2^40'                # count only (3,042,171,781, ~7 s)
./candidates -x '2^71+79'                         # certificate of the discard
./candidates -e -k 34                             # per-level efficiency of each sieve
```

Quote expressions (in zsh, an unquoted `*` is a glob). With the defaults (2^32 sieve, 3^14 sieve,
2^16 × 3^9 joint table), **0.2767% of the integers remain (1 in 361)**; tables are built on each
run in ~1 s.

| option | effect |
|---|---|
| `A [B]` | range [A, B); default A = 2^71, unbounded (20 lines on a terminal) |
| `-q`, `--quantity N` | stop after N candidates |
| `-c`, `--count` | count only (requires B) |
| `-a`, `--random N` | draw N candidates from [A, B) (requires B) |
| `--seed S` | sampling seed |
| `-k`, `--bits K` | 2^K sieve, 1 to 36 (default 32) |
| `-l`, `--three L` | 3^L sieve, 0 to 19 (default 14) |
| `-j`, `--joint J` | 2^min(K,16) × 3^J joint table, 0 to 12 (default 9; 0 disables) |
| `-x`, `--explain n` | certificate for n, any size; repeatable |
| `-e`, `--stats` | live classes and discards (descent, merging, join) per level |
| `-v`, `--verbose` | sieve summary on standard error |
| `-t`, `--threads T` | threads (default: all cores) |
| `--tested [A [B]]` | list tested numbers in order (with `-c`, count) |
| `--descent-only` | classic 2^K sieve, for comparison |
| `--witnesses` | one line per n of [A, B): `n C` or `n E i j m` (T^i(n) = T^j(m)) |

Exit code: 0 success; 1 input error; 4 a certificate failed the simulation check (a bug).

## Rule

If some m with 1 ≤ m < n satisfies T^i(n) = T^j(m) for some i, j ≥ 0, then n is not the smallest
counterexample: m would also be one (cycle or divergence alike). Every sieve is an instance of
this rule:

| case | definition | example |
|---|---|---|
| descent | m = T^i(n) < n | n ≡ 1 (mod 4): T²(n) = (3n+1)/4 < n |
| smaller preimage | T^j(m) = n | n ≡ 2 (mod 3): T((2n−1)/3) = n |
| trajectory merging (Angeltveit 2026) | a predecessor of T^i(n) smaller than n | n = 64q+15: T⁶(n) = T(54q+13) |
| odd–even–even (Hercher 2023) | pattern 1^ℓ00 in x = T^i(n), x < 2n+1 | meets the trajectory of (x−1)/2 |
| join (Roosendaal, Barina) | same class, same steps, smaller residue | m = n − d |

## Sieves

* **2^K sieve.** For n = r + q·2^k, T^k(n) = 3^a·q + T^k(r) (a = odd steps), so x = T^k(n) is known
  mod 3^a. The predecessors of z are 2z and (2z−1)/3 (if z ≡ 2 mod 3, decidable while 3-adic
  precision remains); a predecessor s steps back (t odd) has the form 2^s·3^(a−t)·q + Z and is
  below n for the whole class when 2^s·3^(a−t) ≤ 2^k and the exact inequality holds at the
  class's smallest n. s = 0 is the classic descent. The residue tree is traversed depth-first; a
  discarded class removes all subclasses. **Incremental search** (at level k, only the
  predecessor of x_k other than x_{k−1}) and **exact pruning** (best reachable step balance per
  z mod 3^p) speed up the backward search.
* **3^L sieve.** The same backward search from n itself, using n mod 3^L.
* **Joint table 2^min(K,16) × 3^J.** With n mod 3^J, T^k(n) is known mod 3^(a+J), allowing more odd
  steps back. One row per live class mod 2^16 (1,363), one bit per residue mod 3^J; J = 9 removes
  a further 6.3% in 3.3 MB.

Generation only performs table lookups. All arithmetic is exact: the "+1" term enters the
inequalities, since 3^a < 2^k alone does not guarantee descent (Rozier and Terracol, 2025).

## Efficiency

Surviving 2^K classes for n ≥ 2^71. The Barina and Angeltveit columns come from running the
authors' sieves (Barina's production `h2esieve-24` file and his generator for K = 28; a port of
Angeltveit's step-1 code), as the papers do not report them.

| K | descent only (OEIS A076227) | Barina (production, 2024) | Angeltveit (2026) | **candidates** |
|---|---|---|---|---|
| 24 | 286,581 | 173,181 | 178,304 | **172,868** (1.030%) |
| 28 | 3,524,586 | 1,983,523 | 2,051,228 | **1,976,972** (0.736%) |
| 32 | 41,347,483 | — | 25,005,944 | **23,797,887** (0.554%) |
| 34 | 151,917,636 | — | 89,723,975 | **84,720,792** (0.493%) |
| 36 | 527,682,754 | — | 302,840,108 | **286,848,132** (0.417%) |

The 3^L sieve leaves 55.6% with L = 2 (n mod 9 ∈ {0, 1, 3, 6, 7}), 53.30% with L = 14 and 53.26%
with L = 18. Combined (L = 14, J = 9 unless stated):

| options | remaining | 1 in | tables (Apple M5) | memory |
|---|---|---|---|---|
| `-k 32` (default) | 0.2767% | 361 | 1 s | 0.3 GB |
| `-k 34` | 0.2463% | 406 | 2–4 s | 0.9 GB |
| `-k 36` | 0.2085% | 480 | 8–15 s | 2 GB |
| `-k 32 -j 0` (no joint table) | 0.2953% | 339 | 1 s | 0.3 GB |
| `-k 32 -j 12` | 0.2733% | 366 | 8 s | 0.3 GB |

Throughput: ~20–35 million candidates/s. In a pipeline the verifier is the bottleneck:
`collatz -r -s` reads ~2.5 million numbers/s, covering ~10^9 integers of the range per second.

## Certificates

Each discard has an explicit m; `-x` prints it and checks it by direct simulation:

```
$ ./candidates -x '2^71+79'
n = 2^71+79 = 2361183241434822606927   [72 bits]
✘ DISCARDED: n cannot be the SMALLEST counterexample.
  2^32 sieve, trajectory merging: n ≡ 15 (mod 2^6) fixes the first 6 steps;
  x = T^6(n) is known modulo 3^4, which guarantees a predecessor of x, 1 step
  backwards (1 odd), that is smaller than n.
  m = 1992248359960631574595
  T^6(n) = T^1(m) = 2988372539940947361893   (steps of T)
  Checked by direct simulation: 1 ≤ m < n and the trajectories meet.
  If n were a counterexample, so would be m, and m < n.
```

## Random sampling

```sh
./candidates -a 1000 '2^71' '2^80' | ./collatz -r -s   # 1000 random candidates
./candidates -a 20 --seed 42 '2^71' '2^72'             # reproducible draw
```

`-a N` outputs N distinct untested candidates of [A, B) in ascending order, drawn **uniformly**:
by direct draw of 2^K-table classes when N is small, or by one pass with reservoir sampling when
N ≥ 1/16 of the total or the range is heavily tested. Without `--seed`, the seed used is printed to
standard error. Uniformity is checked by chi-squared tests.

## Use in a search

* The sieves establish only "not the smallest counterexample". To conclude that every n < B
  converges, cover all of [2^71, B) and check that each candidate drops below itself
  (`collatz -s` answers `NO` or `NOTMIN`); `CYCLE` is a counterexample, `UNKNOWN` needs more steps.
* With `-s`, only `NO` is recorded; a `NOTMIN` is regenerated until a run without `-s` takes it to 1.
* The verified limit is 2075·2^60 ≈ 2^71.02 (Barina's site); starting at 2^71 is conservative.
* A < 2^127, B ≤ 2^127. Below 2^71 the sieve is recomputed to hold from A (for tests).

## Not implemented

* **Cycle constraints** (Hercher; Simons and de Weger; Reyes Jiménez 2026): silent on divergent
  trajectories, so they discard no n.
* **"Recursive sufficiency"** (Ansari, *Notes on Number Theory and Discrete Mathematics* 31(3),
  2025): its induction lacks justification; e.g. the class 36k+27 has no smaller preimage and
  contains every 2^(6t) − 1, which rises for 6t steps.
* **Angeltveit's full algorithm:** step 2 applies four sieves to the 24 steps after T^(N−A)(n)
  with bit vectors (estimated under 0.02% of n < 2^72 left, versus 0.2767% here, where the table
  must fit in memory). It uses each full n, so it belongs in a combined generate-and-verify mode
  of the verifier.
* **Joint table at full depth:** the gain is in shallow levels; it would need a per-candidate
  search, costlier than verification.

## Generator tests

`tests/ref_candidates.py` reimplements the sieves naively. `tests/test_candidates.py`
(`make test-candidates`, ~1 min) checks:

* every discard in the tested windows (~40 thousand) by step-by-step simulation;
* survivors and per-level counts against the reference, known classes (mod 32, 9, 288) and OEIS
  A076227, windows up to 2^127, independence from pruning and thread count;
* `-a`: distinct, ordered, in range, untested, reproducible, uniform (chi-squared);
* the record: only `NO` in [2^71, 2^127) is stored, concurrently and under interruption (Ctrl+C,
  `kill`, closed terminal, `| head`), with output ending on a complete line and `tested.bin` in
  order; an independent reader (`tests/tested_fmt.py`) checks order, uniqueness and CRCs;
  corrupted data never excludes anything; each `data.csv` line is checked by simulation.
