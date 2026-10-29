#!/usr/bin/env python3
"""Convert the trajectory data of Tested/ (data.csv and data journals) into an SQLite database.

Usage: python3 tools/data_to_sqlite.py [DIR] [-o OUT]

DIR defaults to $COLLATZ_TESTED, or Tested/ at the repository root (next to the executables).
OUT defaults to DIR/data.sqlite. The database is rebuilt from scratch on every run, written to a
temporary file and renamed atomically, so a reader never sees a partial database.

Table `data`, one row per n, inserted in ascending order of n (rowid order = numeric order):
  n, x_m              TEXT     exact decimal values (they exceed SQLite's 64-bit integers)
  bits, x_m_bits      INTEGER  bit lengths of n and x_m
  k, odd_k, j_minus_k, odd_j_minus_k, exponent_j, m, odd_m
                      INTEGER  the data.csv columns (see README)
  total               INTEGER  total stopping time δ(n) = k + j_minus_k + exponent_j

Lines not yet merged into data.csv (data.journal.csv, data.journal.<pid>.csv) are included.
Invalid lines (e.g. a truncated last line of a killed process) are skipped with a warning.
"""

import argparse
import os
import re
import sqlite3
import sys
import time

MIN, LIMIT = 1 << 71, 1 << 127          # range recorded by collatz (as in tested.h)
COLS = 9                                # n,k,odd_k,j_minus_k,odd_j_minus_k,exponent_j,m,x_m,odd_m
JOURNAL = re.compile(r"data\.journal(\.\d+)?\.csv$")
FIELD = re.compile(r"\d+$")
SIGNED = re.compile(r"-?\d+$")          # only j_minus_k may be negative

SCHEMA = """
CREATE TABLE data (
    n             TEXT    NOT NULL,
    bits          INTEGER NOT NULL,
    k             INTEGER NOT NULL,
    odd_k         INTEGER NOT NULL,
    j_minus_k     INTEGER NOT NULL,
    odd_j_minus_k INTEGER NOT NULL,
    exponent_j    INTEGER NOT NULL,
    total         INTEGER NOT NULL,
    m             INTEGER NOT NULL,
    x_m           TEXT    NOT NULL,
    x_m_bits      INTEGER NOT NULL,
    odd_m         INTEGER NOT NULL
);
"""


def default_dir():
    env = os.environ.get("COLLATZ_TESTED")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Tested")


def parse(line):
    """Fields of a valid data line as integers, or None."""
    f = line.rstrip("\n").split(",")
    if len(f) != COLS or not all(SIGNED.match(x) if i == 3 else FIELD.match(x) for i, x in enumerate(f)):
        return None
    v = [int(x) for x in f]
    return v if MIN <= v[0] < LIMIT else None


def read_rows(d):
    """Valid rows of data.csv and the data journals, by n (a repeated n is kept once)."""
    names = ["data.csv"] + sorted(nm for nm in os.listdir(d) if JOURNAL.match(nm))
    rows, bad, used = {}, 0, []
    for nm in names:
        path = os.path.join(d, nm)
        if not os.path.isfile(path):
            continue
        used.append(nm)
        with open(path, encoding="ascii", errors="replace") as fh:
            for i, line in enumerate(fh):
                v = parse(line)
                if v is None:
                    bad += not (i == 0 and line.startswith("n,"))   # the header is not an error
                    continue
                rows.setdefault(v[0], v)
    return rows, bad, used


def write_db(rows, out):
    tmp = out + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    db = sqlite3.connect(tmp)
    db.execute("PRAGMA journal_mode = OFF")
    db.execute("PRAGMA synchronous = OFF")
    db.executescript(SCHEMA)
    db.executemany(
        "INSERT INTO data VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ((str(n), n.bit_length(), k, ok, jk, ojk, e, k + jk + e, m, str(xm), xm.bit_length(), om)
         for n, k, ok, jk, ojk, e, m, xm, om in (rows[n] for n in sorted(rows))))
    db.commit()
    db.close()
    os.replace(tmp, out)


def main():
    ap = argparse.ArgumentParser(description="Convert Tested/data.csv into an SQLite database.")
    ap.add_argument("dir", nargs="?", default=None, help="data directory (default: $COLLATZ_TESTED or Tested/)")
    ap.add_argument("-o", "--output", help="database file (default: DIR/data.sqlite)")
    a = ap.parse_args()
    d = a.dir or default_dir()
    out = a.output or os.path.join(d, "data.sqlite")
    if not os.path.isdir(d):
        sys.exit(f"data_to_sqlite: directory not found: {d}")
    t0 = time.time()
    rows, bad, used = read_rows(d)
    if not used:
        sys.exit(f"data_to_sqlite: no data.csv or data journal in {d} (run collatz with --data)")
    if bad:
        print(f"data_to_sqlite: warning: {bad:,} invalid line(s) skipped", file=sys.stderr)
    write_db(rows, out)
    size = os.path.getsize(out) / 1e6
    print(f"{len(rows):,} rows from {', '.join(used)} -> {out} ({size:.1f} MB, {time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
