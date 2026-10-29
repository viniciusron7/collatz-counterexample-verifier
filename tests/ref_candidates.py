#!/usr/bin/env python3
"""
Independent — and deliberately simple — reference for the `candidates` generator.

Same mathematics, naive implementation: no incremental search, no table-based pruning, no
128-bit arithmetic. Accelerated map T(x) = x/2 (even) or (3x+1)/2 (odd).

Rule: n is discarded if there is m with 1 ≤ m < n and T^i(n) = T^j(m). The sieves look for such
an m for whole classes of n:

  2^K sieve  for the class n = r + q·2^k (k = 1..K, n ≥ S), the first k steps depend only on
             r: T^k(n) = 3^a·q + T^k(r), with a = number of odd steps. Starting from
             x = T^k(n), look for a predecessor z (T^s(z) = x) that is smaller than n for EVERY
             n of the class. The backward search takes odd steps only while x mod 3^a
             guarantees the right parity. s = 0 is descent (z = x).
  3^L sieve  the same search from n itself, using n mod 3^L.

Direct use:  python3 tests/ref_candidates.py K L        (prints the counts)
"""

import sys

VERIFIED_LIMIT = 2 ** 71


def T(x):
    return x // 2 if x % 2 == 0 else (3 * x + 1) // 2


def T_iter(x, k):
    for _ in range(k):
        x = T(x)
    return x


def qmin_of(S, R, M):
    """Smallest q ≥ 0 with R + q·M ≥ S."""
    return max(0, -((R - S) // M))


def back_search(M, R, A, Y, qmin):
    """Class n = R + q·M (q ≥ qmin). A point of the trajectory of n is x = 3^A·q + Y.
    Looks for z with T^s(z) = x and z < n for every q ≥ qmin. Each candidate has the form
    z = 2^s·3^(A−t)·q + Z (s steps backwards, t of them odd). Returns (s, t, Z) or None."""
    stack = [(0, 0, Y)]
    while stack:
        s, t, Z = stack.pop()
        coef = 2 ** s * 3 ** (A - t)
        if coef == M and Z < R:
            return (s, t, Z)
        if coef < M and (M - coef) * qmin > Z - R:
            return (s, t, Z)
        if t == A:            # x mod 3 unknown from here backwards: only even steps, which only grow
            continue
        if Z % 3 == 0:        # a multiple of 3 has no odd predecessor
            continue
        # the smallest reachable coefficient spends all remaining precision on odd steps
        if 2 ** (s + 1 + A - t) <= M:
            stack.append((s + 1, t, 2 * Z))
        if Z % 3 == 2 and 2 ** (s + A - t) <= M:
            stack.append((s + 1, t + 1, (2 * Z - 1) // 3))
    return None


def class_info(r, k):
    """(a, y): number of odd steps and T^k(r), simulating r step by step."""
    a, x = 0, r
    for _ in range(k):
        if x % 2:
            a += 1
        x = T(x)
    return a, x


def test_class2(r, k, S):
    """Does the class r (mod 2^k), n ≥ S, have a certificate at level k? Returns (s, t, Z, a) or None."""
    a, y = class_info(r, k)
    w = back_search(2 ** k, r, a, y, qmin_of(S, r, 2 ** k))
    return None if w is None else w + (a,)


def sieve2(K, S):
    """Surviving classes mod 2^K (a set) and the count of live classes per level."""
    alive = [0] * (K + 1)
    out = set()
    stack = [(0, 0)]
    while stack:
        k, r = stack.pop()
        if k >= 1 and test_class2(r, k, S) is not None:
            continue
        alive[k] += 1
        if k == K:
            out.add(r)
            continue
        stack.append((k + 1, r))
        stack.append((k + 1, r + 2 ** k))
    return out, alive


def test_class3(nu, L, S):
    if L == 0:
        return None
    M = 3 ** L
    return back_search(M, nu, L, nu, qmin_of(S, nu, M))


def sieve3(L, S):
    return {nu for nu in range(3 ** L) if test_class3(nu, L, S) is None}


def joint_table(K0, J, S):
    """Joint table: for each class r mod 2^K0 that survives the 2-adic sieve, the set of
    ν mod 3^J for which no certificate exists — neither from n (plain 3^J sieve) nor from
    T^k(n), k = 1..K0, knowing n mod 2^k AND n mod 3^J. Full search at every level (without the
    incremental shortcut of the C code)."""
    if J == 0:
        return None
    m3 = 3 ** J
    pure = sieve3(J, S)
    s2, _ = sieve2(K0, S)
    info = {}
    for k in range(1, K0 + 1):
        for r in s2:
            rk = r % 2 ** k
            if (k, rk) not in info:
                info[(k, rk)] = class_info(rk, k)
    table = {}
    for r in s2:
        alive = set()
        for nu in pure:
            for k in range(1, K0 + 1):
                rk = r % 2 ** k
                a, y = info[(k, rk)]
                q0 = (nu - rk) * pow(2, -k, m3) % m3          # n = rk + q·2^k with q ≡ q0 (mod 3^J)
                M = 2 ** k * m3
                R = rk + q0 * 2 ** k
                if back_search(M, R, a + J, 3 ** a * q0 + y, qmin_of(S, R, M)) is not None:
                    break
            else:
                alive.add(nu)
        table[r] = alive
    return table


def candidates(lo, hi, K, L, J, S, k0max=16):
    """Candidates in [lo, hi) according to the reference (same tables as the C code)."""
    s2, _ = sieve2(K, S)
    s3 = sieve3(L, S)
    k0 = min(K, k0max)
    jt = joint_table(k0, J, S)
    out = []
    for n in range(lo, hi):
        if n % 2 ** K not in s2 or n % 3 ** L not in s3:
            continue
        if jt is not None and n % 3 ** J not in jt.get(n % 2 ** k0, ()):
            continue
        out.append(n)
    return out


def descent_only_counts(K):
    """Classes mod 2^k with T^j(r) ≥ r for j ≤ k in the class sense (OEIS A076227):
    computed separately, with the descent inequality only, to check --descent-only."""
    S = VERIFIED_LIMIT
    alive = [0] * (K + 1)
    stack = [(0, 0)]
    while stack:
        k, r = stack.pop()
        if k >= 1:
            a, y = class_info(r, k)
            M = 2 ** k
            if 3 ** a < M and (M - 3 ** a) * qmin_of(S, r, M) > y - r:
                continue
        alive[k] += 1
        if k < K:
            stack.append((k + 1, r))
            stack.append((k + 1, r + 2 ** k))
    return alive


def check_witness(n, i, j, m):
    """Does the certificate prove that n is not the smallest counterexample? (1 ≤ m < n and T^i(n) = T^j(m))"""
    return 1 <= m < n and T_iter(n, i) == T_iter(m, j)


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    L = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    s2, alive = sieve2(K, VERIFIED_LIMIT)
    s3 = sieve3(L, VERIFIED_LIMIT)
    print("live per level:", alive[1:])
    print(f"2^{K}: {len(s2)} classes;  3^{L}: {len(s3)} of {3 ** L}")
