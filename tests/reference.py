#!/usr/bin/env python3
"""
Independent — and deliberately naive — reference for the `collatz` verifier.

It simulates the Collatz map ONE STEP AT A TIME, with Python integers:

    C(x) = x / 2      if x is even
    C(x) = 3·x + 1    if x is odd

and records the index k (in C steps) at which each event first happens:

    sigma   first k ≥ 1 with x_k < n        (stopping time; does not exist for n = 1)
    bound   first k ≥ 0 with x_k < 2^71     (0 if n < 2^71)
    total   first k ≥ 0 with x_k = 1        (total stopping time δ(n))

and also sigmaT / totalT: how many of those k steps were divisions by 2. This is the
number of steps of the accelerated map T(x) = x/2 or (3x+1)/2, because each T step
ends with exactly one division by 2.

The value of this file lies in being obviously correct, not fast: no runs,
blocks, 2-adic arithmetic or shortcuts.

Direct use:  python3 tests/reference.py 27 "2**71+1"
"""

import sys

VERIFIED_BITS = 71
VERIFIED_LIMIT = 2 ** VERIFIED_BITS   # every integer in [1, 2^71) has already been verified (Barina, 2025)

FIELDS = ("sigma", "sigmaT", "bound", "total", "totalT")


def events(n, max_steps=None, stop_at_sigma=False):
    """Follow the trajectory of n until x_k = 1.

    Return {"sigma", "sigmaT", "bound", "total", "totalT"} (None = did not happen)
    plus "steps" (how many C steps were simulated).
      max_steps:     stop after max_steps steps (later events stay None);
      stop_at_sigma: stop right after finding sigma (bound/total may stay None)."""
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    ev = dict.fromkeys(FIELDS)
    x = n
    k = 0          # C steps taken so far
    halvings = 0   # how many of them were x -> x/2  (= T steps)

    # Step 0: x_0 = n.  (sigma requires k ≥ 1, and x_0 < n is false anyway.)
    if x < VERIFIED_LIMIT:
        ev["bound"] = 0
    if x == 1:
        ev["total"], ev["totalT"] = 0, 0

    while x != 1:
        if max_steps is not None and k >= max_steps:
            break
        if stop_at_sigma and ev["sigma"] is not None:
            break
        if x % 2 == 0:
            x = x // 2
            halvings += 1
        else:
            x = 3 * x + 1
        k += 1
        if ev["sigma"] is None and x < n:
            ev["sigma"], ev["sigmaT"] = k, halvings
        if ev["bound"] is None and x < VERIFIED_LIMIT:
            ev["bound"] = k
        if x == 1:
            ev["total"], ev["totalT"] = k, halvings
    ev["steps"] = k
    return ev


def expected_line(n, stop=False, limit=None, ev=None):
    """Fields that `collatz -r` must print for n (all except 'time').

    stop  = True for the -s mode (stops at sigma).
    limit = N of --max-steps (None = no limit): an event appears only if its
            index k is ≤ N, and the program never takes more than N C steps.
    ev    = an already computed result of events(n) (optional, to avoid recomputing).

    Return a dict with 'verdict', the 5 fields (string: number or '-') and 'bits'."""
    if ev is None:
        ev = events(n, stop_at_sigma=stop, max_steps=limit)
    # ev must cover everything the program can see: down to 1, up to the limit or (with -s) up to sigma.
    covered = (ev["total"] is not None
               or (limit is not None and ev["steps"] >= limit)
               or (stop and ev["sigma"] is not None))
    if not covered:
        raise ValueError("events(n) does not cover the required stretch of the trajectory")

    def seen(key):
        k = ev[key]
        return k is not None and (limit is None or k <= limit)

    show_sigma = seen("sigma")
    if stop:
        # -s stops at sigma, so bound appears only if it is already proved by then: 0 (n < 2^71)
        # or sigma (x_sigma < 2^71; before sigma every x_k ≥ n ≥ 2^71, hence bound ≥ sigma).
        show_bound = seen("bound") and (ev["bound"] == 0 or (show_sigma and ev["bound"] <= ev["sigma"]))
        show_total = (n == 1)          # -s with n = 1: the only event is x_0 = 1
    else:
        show_bound = seen("bound")
        show_total = seen("total")

    if show_total or show_bound:
        verdict = "NO"                 # reached 1, or dropped below 2^71 (or n < 2^71)
    elif show_sigma:
        verdict = "NOTMIN"             # fell below n, but without a complete proof
    else:
        verdict = "UNKNOWN"            # the limit ran out before any conclusion

    def val(key, show):
        return str(ev[key]) if show else "-"

    return {
        "verdict": verdict,
        "sigma": val("sigma", show_sigma),
        "sigmaT": val("sigmaT", show_sigma),
        "bound": val("bound", show_bound),
        "total": val("total", show_total),
        "totalT": val("totalT", show_total),
        "bits": str(n.bit_length()),
    }


def format_line(d):
    """Dict from expected_line() -> text in the -r output format (without 'time')."""
    return "{verdict} sigma={sigma} sigmaT={sigmaT} bound={bound} total={total} " \
           "totalT={totalT} bits={bits}".format(**d)


def _eval_arith(text):
    """Evaluate a Python arithmetic expression ("2**71+1", "(10**50-1)//9")."""
    import ast
    import operator as op
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.FloorDiv: op.floordiv,
           ast.Mod: op.mod, ast.Pow: op.pow, ast.USub: op.neg, ast.UAdd: op.pos}

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression: " + text)
    return ev(ast.parse(text, mode="eval").body)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        value = _eval_arith(arg)
        print(arg, "->", format_line(expected_line(value)))
        print(" " * len(arg), "   -s:", format_line(expected_line(value, stop=True)))
