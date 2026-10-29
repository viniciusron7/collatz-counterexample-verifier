/*
 * candidates.c — generator of candidates for the SMALLEST counterexample to the Collatz conjecture.
 *
 * If a counterexample exists, a smallest one, N, exists. Every integer below 2^71 has already
 * been verified (D. Barina, 2025), so N ≥ 2^71. This program lists the n of an interval [A, B)
 * that CAN STILL be that N; all others are discarded by sieves, without simulating the
 * trajectory of each n. Guarantee: if the smallest counterexample lies in [A, B), it is in the
 * list. (A counterexample that is not the smallest, such as 2N, may be left out; this does not
 * matter, since finding N suffices.)
 *
 * Accelerated map: T(x) = x/2 (x even) or (3x+1)/2 (x odd). All steps below are steps of T.
 *
 * SINGLE RULE. n is discarded when there exist m with 1 ≤ m < n and i, j ≥ 0 such that
 *     T^i(n) = T^j(m)        (the trajectories of n and m meet).
 * If n were a counterexample, so would be the meeting point (its trajectory does not reach 1),
 * and then so would m: n cannot be the SMALLEST. Special cases:
 *   • descent (j = 0, m = T^i(n) < n): the classical 2^k sieve;
 *   • smaller preimage (i = 0): e.g. n ≡ 2 (mod 3) ⇒ T((2n−1)/3) = n; n ≡ 4 (mod 9) ⇒
 *     T^3((8n−5)/9) = n: the 3^k sieve;
 *   • trajectory merging (path merging, Angeltveit 2026), which includes the odd–even–even
 *     lemma of Hercher (2023) and the residue "joining" of Barina (2021).
 *
 * 2^K SIEVE. For n = r + q·2^k (q free), the first k steps depend only on r:
 *     T^k(r + q·2^k) = 3^a·q + y,    a = number of odd steps, y = T^k(r).
 * Hence x = T^k(n) ≡ y (mod 3^a): the odd steps reveal x modulo 3^a. Walking backwards from x
 * — predecessors 2z (always) and (2z−1)/3 (if z ≡ 2 mod 3, which is known only while 3-adic
 * precision remains) — every predecessor has the form
 *     z = 2^s·3^(a−t)·q + Z      (s steps backwards, t of them odd; T^s(z) = x),
 * and z < n holds for the whole class if 2^s·3^(a−t) ≤ 2^k and the exact inequality holds at the
 * smallest q of the class (or, with equal coefficients, if Z < r). s = 0 is plain descent. The
 * residue tree (r mod 2, mod 4, ...) is traversed depth-first; a discarded class takes all its
 * subclasses with it. Incremental search: at level k it suffices to explore the predecessor of
 * x_k that is NOT x_{k−1}; the rest of the backward tree was already examined at the previous
 * level.
 *
 * 3^L SIEVE. The same backward search, from n itself, using n mod 3^L.
 *
 * JOINT TABLE 2^min(K,16) × 3^J (-j, default J = 9; 0 disables). Knowing also n mod 3^J,
 * x = T^k(n) becomes known modulo 3^(a+J), and the backward search can take more odd steps.
 * For each live class mod 2^min(K,16), the table stores the residues mod 3^J that survive.
 *
 * n is a candidate if it survives all three tables (2^K, 3^L and joint). All arithmetic is
 * exact (the term y, coming from the "+1"s, enters the inequalities: 3^a < 2^k alone does not
 * guarantee descent — Rozier and Terracol, 2025). Every discard has an explicit certificate m:
 * see --explain and --witnesses.
 *
 * TESTED. The numbers that collatz has already proven not to be counterexamples are kept in
 * Tested/tested.bin, in order (collatz writes them; format in tested.h); the generator does not
 * list them again, and --tested prints the contents as text.
 *
 * Typical use:   ./candidates '2^71' '2^71+10^9' | ./collatz -r -s
 */

#include <stdio.h>
#include <gmp.h>
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "tested.h"            /* format of the Tested/ directory (shared with collatz) */

__extension__ typedef unsigned __int128 u128;
__extension__ typedef __int128 s128;

#define VERIFIED_BITS 71        /* every integer in [1, 2^71) converges (Barina, 2025) */
#define K_MAX         36        /* 2^K sieve: up to 2^36 (the table takes 4 bytes per class) */
#define K_DEF         32
#define L_MAX         19
#define L_DEF         14
#define SPLIT_K       14        /* level at which the 2^K tree is split among the threads */
#define NODE_CAP      2000000   /* nodes per backward search; on overflow it merely stops discarding */
#define U128_MAX      (~(u128)0)
#define N_LIMIT       ((u128)1 << 127)   /* A and B must be smaller than 2^127 */

static int spawn_thread(pthread_t *th, void *(*fn)(void *), void *arg);   /* see below */

/* ------------------------------------------------------------------------- */
/* Utilities                                                                 */

static u128 P3[81];                 /* 3^i, i ≤ 80 */
static const u128 INV3 = (((u128)0xAAAAAAAAAAAAAAAAULL) << 64) | 0xAAAAAAAAAAAAAAABULL;  /* 3·INV3 ≡ 1 (mod 2^128) */

static void init_tables(void)
{
    P3[0] = 1;
    for (int i = 1; i <= 80; i++) P3[i] = P3[i - 1] * 3;
}

/* z mod 3 without 128-bit division (2^64 ≡ 1 mod 3). */
static inline unsigned mod3(u128 z)
{
    uint64_t lo = (uint64_t)z, hi = (uint64_t)(z >> 64);
    return (unsigned)((lo % 3 + hi % 3) % 3);
}

/* z/3 for z a multiple of 3 (exact division: multiply by the inverse of 3 mod 2^128). */
static inline u128 div3_exact(u128 z) { return z * INV3; }

/* Pruning of the backward search. A path with e even and o odd steps, starting at a node with
   coefficient 2^s·3^p (p = remaining 3-adic precision), ends with coefficient
   2^(s+e+o)·3^(p−o): in log2, s + p·log2(3) + (e − λ·o), with λ = log2(3) − 1. GTAB[p][z mod 3^p]
   is the smallest value of (e − λ·o) over all valid paths (including the empty one), computed
   exactly by dynamic programming. If not even the best path reaches a coefficient ≤ M, the node
   is pruned. Pruning only cuts dead-end branches: it changes the running time, not the result. */
#define GP0 12
static float   *GTAB[GP0 + 1];
static uint32_t P3_32[GP0 + 1];
static double   LOG2_3, LAMBDA;
static int      G_no_prune = 0;      /* --no-prune (tests) */

static void init_gtab(void)
{
    LOG2_3 = log2(3.0);
    LAMBDA = LOG2_3 - 1.0;
    for (int p = 0; p <= GP0; p++) {
        uint32_t m = 1;
        for (int i = 0; i < p; i++) m *= 3;
        P3_32[p] = m;
        GTAB[p] = malloc((size_t)m * sizeof(float));
        if (!GTAB[p]) { fprintf(stderr, "candidates: out of memory\n"); exit(1); }
        if (p == 0) { GTAB[0][0] = 0.0f; continue; }
        /* c(σ): best cost starting with an odd step at σ (or stopping right away). */
        double *c = malloc((size_t)m * sizeof *c);
        if (!c) { fprintf(stderr, "candidates: out of memory\n"); exit(1); }
        uint32_t m1 = m / 3;
        for (uint32_t z = 0; z < m; z++) {
            double best = 0.0;
            if (z % 3 == 2) {
                uint32_t nz = (uint32_t)(((uint64_t)2 * z - 1) / 3 % m1);
                double v = -LAMBDA + GTAB[p - 1][nz];
                if (v < best) best = v;
            }
            c[z] = best;
        }
        /* g(z) = min_j ( j + c(2^j·z) ): j even steps first. j > λ·p never pays off. */
        int J = (int)ceil(LAMBDA * p) + 2;
        for (uint32_t z = 0; z < m; z++) {
            double best = c[z];
            if (z % 3) {
                uint64_t w = z;
                for (int j = 1; j <= J; j++) {
                    w = 2 * w % m;
                    double v = j + c[w];
                    if (v < best) best = v;
                }
            }
            GTAB[p][z] = (float)best;
        }
        free(c);
    }
}

/* z mod 3^p for p ≤ GP0 (3^12 < 2^20), without 128-bit division. */
static inline uint32_t mod3p(u128 z, int p)
{
    uint64_t m = P3_32[p], lo = (uint64_t)z, hi = (uint64_t)(z >> 64);
    uint64_t r64 = ((uint64_t)1 << 32) % m;            /* 2^64 mod m = (2^32 mod m)^2 mod m */
    r64 = r64 * r64 % m;
    return (uint32_t)(((hi % m) * r64 + lo % m) % m);
}

static int floor_log2_u128(u128 v)   /* v ≥ 1 */
{
    uint64_t hi = (uint64_t)(v >> 64);
    return hi ? 127 - __builtin_clzll(hi) : 63 - __builtin_clzll((uint64_t)v);
}

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Writes v in decimal ending at end; returns the start. */
static char *u128_dec(char *end, u128 v)
{
    char *p = end;
    if (v >> 64) {
        const uint64_t P19 = 10000000000000000000ULL;
        uint64_t lo = (uint64_t)(v % P19);
        v /= P19;
        for (int i = 0; i < 19; i++) { *--p = (char)('0' + lo % 10); lo /= 10; }
        if (!v) return p;
    }
    uint64_t w = (uint64_t)v;
    do { *--p = (char)('0' + w % 10); w /= 10; } while (w);
    return p;
}

static void fmt_u64(char *buf, uint64_t v)   /* 1234567 → "1,234,567" */
{
    char tmp[32];
    int n = snprintf(tmp, sizeof tmp, "%" PRIu64, v), j = 0;
    for (int i = 0; i < n; i++) {
        if (i && (n - i) % 3 == 0) buf[j++] = ',';
        buf[j++] = tmp[i];
    }
    buf[j] = 0;
}

static void fmt_pct(char *buf, size_t sz, double frac, int dec)   /* 0.5541% */
{
    snprintf(buf, sz, "%.*f%%", dec, 100.0 * frac);
}

static int count_cpus(void)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n < 1 ? 1 : n > 64 ? 64 : (int)n;
}

/* ------------------------------------------------------------------------- */
/* Backward search (the core of both sieves)                                 */

/* Discard certificate for the class n ≡ R (mod M), M = 2^k·3^Lw (R < M):
   m = 2^s·3^(A−t)·q + Z, with q = ⌊n/M⌋, satisfies 1 ≤ m < n and T^s(m) = T^k(n).
     2^K sieve:        Lw = 0,  A = a (number of odd steps among the first k);
     3^L sieve:        k = 0,   Lw = A = L;
     joint table:      Lw = J,  A = a + J. */
typedef struct {
    int  k, Lw, s, t, A;
    u128 Z;
} witness_t;

/* Is z = 2^s·3^(A−t)·q + Z smaller than n = R + q·M for every q ≥ qmin? */
static inline int below_n(u128 M, u128 R, int s, int t, int A, u128 Z, u128 qmin)
{
    u128 coef = ((u128)1 << s) * P3[A - t];
    if (coef > M) return 0;
    if (coef == M) return Z < R;                          /* m = n − (R − Z) */
    return (s128)((M - coef) * qmin) > (s128)Z - (s128)R;   /* holds at the smallest q, hence at all */
}

/* Depth-first search over the predecessors of x = 3^A·q + Y, from the node (s0, t0, Z0).
   maxe = ⌊log2 M⌋: after spending all remaining precision on odd steps, the coefficient would
   be 2^(s + A − t); if that exceeds M, no descendant qualifies. */
static int back_search(u128 M, int maxe, double logM, u128 R, int A, int s0, int t0, u128 Z0, u128 qmin, witness_t *w)
{
    struct { int s, t; u128 Z; } st[2 * 128 + 8];
    if (s0 + A - t0 > maxe) return 0;
    int sp = 0;
    st[sp].s = s0; st[sp].t = t0; st[sp].Z = Z0; sp++;
    long nodes = 0;
    const double lim = logM + 1e-4;     /* margin ≫ rounding error; affects only pruning */
    while (sp) {
        sp--;
        int s = st[sp].s, t = st[sp].t;
        u128 Z = st[sp].Z;
        if (++nodes > NODE_CAP) return 0;
        if (!G_no_prune) {
            int p = A - t;
            double bound = s + p * LOG2_3;
            if (p <= GP0) bound += GTAB[p][mod3p(Z, p)];
            else          bound += GTAB[GP0][mod3p(Z, GP0)] - LAMBDA * (p - GP0);
            if (bound > lim) continue;      /* not even the best path from here reaches coefficient ≤ M */
        }
        if (below_n(M, R, s, t, A, Z, qmin)) {
            w->s = s; w->t = t; w->A = A; w->Z = Z;
            return 1;
        }
        if (t == A) continue;           /* no precision left for odd steps: 2z, 4z, ... only grow */
        unsigned zm = mod3(Z);          /* z ≡ Z (mod 3), since A − t ≥ 1 */
        if (zm == 0) continue;          /* multiple of 3: never an odd predecessor */
        if (s + 1 + A - t <= maxe) {    /* even predecessor: 2z */
            st[sp].s = s + 1; st[sp].t = t; st[sp].Z = 2 * Z; sp++;
        }
        if (zm == 2) {                  /* odd predecessor: (2z − 1)/3 */
            st[sp].s = s + 1; st[sp].t = t + 1; st[sp].Z = div3_exact(2 * Z - 1); sp++;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------------- */
/* 2^K sieve                                                                 */

static int  G_K = K_DEF, G_L = L_DEF;
static int  G_only_descent = 0;      /* --descent-only */
static u128 G_S;                     /* the sieve is valid for every n ≥ G_S */

/* Class n = r + q·2^k (n ≥ S), x_k = T^k(n) = 3^a·q + y; last_odd = step k−1 → k was odd.
   Returns 1 (with the certificate) if the whole class can be discarded. */
static int test2(int k, uint64_t r, int a, u128 y, int last_odd, u128 S, witness_t *w)
{
    u128 M = (u128)1 << k;
    u128 qmin = S > r ? (S - r + M - 1) >> k : 0;
    w->k = k; w->Lw = 0;
    if (below_n(M, r, 0, 0, a, y, qmin)) {        /* descent: m = x_k */
        w->s = 0; w->t = 0; w->A = a; w->Z = y;
        return 1;
    }
    if (G_only_descent) return 0;
    if (last_odd)                                  /* x_{k−1} = (2x_k − 1)/3; the other predecessor is 2x_k */
        return back_search(M, k, (double)k, r, a, 1, 0, 2 * y, qmin, w);
    if (a >= 1 && mod3(y) == 2)                    /* x_{k−1} = 2x_k; the other one is (2x_k − 1)/3 */
        return back_search(M, k, (double)k, r, a, 1, 1, div3_exact(2 * y - 1), qmin, w);
    return 0;
}

typedef struct { uint64_t r; u128 y; int a, last_odd; } node2_t;

/* Surviving classes (r < 2^K ≤ 2^36) are stored in 4 bytes: the low 32 bits, in one of 16
   "buckets" selected by bits 32 to 35. */
#define NBUCK 16
_Static_assert(K_MAX <= 36, "the buckets cover only r < 2^36");

typedef struct {
    uint32_t *v[NBUCK]; size_t n[NBUCK], cap[NBUCK];   /* surviving classes at level K */
    node2_t  *fr; size_t nfr, capfr;        /* frontier (sequential phase only) */
    uint64_t alive[K_MAX + 1], desc[K_MAX + 1], merge[K_MAX + 1], join[K_MAX + 1];
    int      stop;                          /* level at which the sequential phase stops */
} acc2_t;

static void *xrealloc(void *p, size_t sz)
{
    void *q = realloc(p, sz ? sz : 1);
    if (!q) {
        fprintf(stderr, "candidates: out of memory (%zu bytes)\n", sz);
        exit(1);
    }
    return q;
}

static void push_res(acc2_t *acc, uint64_t r)
{
    int b = (int)(r >> 32);
    if (acc->n[b] == acc->cap[b]) {
        acc->cap[b] = acc->cap[b] ? acc->cap[b] + acc->cap[b] / 2 : 1 << 14;
        acc->v[b] = xrealloc(acc->v[b], acc->cap[b] * sizeof *acc->v[b]);
    }
    acc->v[b][acc->n[b]++] = (uint32_t)r;
}

static void classify(acc2_t *acc, const witness_t *w)
{
    if (w->s == 0) acc->desc[w->k]++;
    else if (((u128)1 << w->s) * P3[w->A - w->t] == ((u128)1 << w->k)) acc->join[w->k]++;
    else acc->merge[w->k]++;
}

static void dfs2(int k, uint64_t r, int a, u128 y, int last_odd, acc2_t *acc)
{
    if (k >= 1) {
        witness_t w;
        if (test2(k, r, a, y, last_odd, G_S, &w)) {
            classify(acc, &w);
            return;
        }
    }
    acc->alive[k]++;
    if (k == acc->stop) {
        if (acc->stop == G_K) {
            push_res(acc, r);
        } else {
            if (acc->nfr == acc->capfr) {
                acc->capfr = acc->capfr ? 2 * acc->capfr : 1024;
                acc->fr = xrealloc(acc->fr, acc->capfr * sizeof *acc->fr);
            }
            acc->fr[acc->nfr++] = (node2_t){ r, y, a, last_odd };
        }
        return;
    }
    for (int b = 0; b < 2; b++) {
        uint64_t rr = r | ((uint64_t)b << k);
        u128 yy = b ? y + P3[a] : y;                /* x_k for bit k of n equal to b */
        if (yy & 1) dfs2(k + 1, rr, a + 1, (3 * yy + 1) >> 1, 1, acc);
        else        dfs2(k + 1, rr, a, yy >> 1, 0, acc);
    }
}

static struct {
    const node2_t *fr;
    size_t         nfr;
    atomic_size_t  next;
    int            split;
} W2;

static void *worker2(void *arg)
{
    acc2_t *acc = arg;
    acc->stop = G_K;
    for (;;) {
        size_t i = atomic_fetch_add(&W2.next, 1);
        if (i >= W2.nfr) break;
        const node2_t *nd = &W2.fr[i];
        int k = W2.split;
        for (int b = 0; b < 2; b++) {
            uint64_t rr = nd->r | ((uint64_t)b << k);
            u128 yy = b ? nd->y + P3[nd->a] : nd->y;
            if (yy & 1) dfs2(k + 1, rr, nd->a + 1, (3 * yy + 1) >> 1, 1, acc);
            else        dfs2(k + 1, rr, nd->a, yy >> 1, 0, acc);
        }
    }
    return NULL;
}

/* Sorts v[0..n) by radix sort on two 16-bit digits (tmp has room for n). */
static void radix_sort32(uint32_t *v, size_t n, uint32_t *tmp)
{
    size_t *cnt = xrealloc(NULL, 65536 * sizeof *cnt);
    uint32_t *src = v, *dst = tmp;
    for (int sh = 0; sh < 32; sh += 16) {
        memset(cnt, 0, 65536 * sizeof *cnt);
        for (size_t i = 0; i < n; i++) cnt[(src[i] >> sh) & 0xFFFF]++;
        size_t sum = 0;
        for (int d = 0; d < 65536; d++) { size_t c = cnt[d]; cnt[d] = sum; sum += c; }
        for (size_t i = 0; i < n; i++) dst[cnt[(src[i] >> sh) & 0xFFFF]++] = src[i];
        uint32_t *t = src; src = dst; dst = t;
    }
    free(cnt);                               /* two passes: the result ends up back in v */
}

typedef struct {
    uint32_t *res;  size_t n;                /* low 32 bits of the classes, in increasing order */
    size_t    off[NBUCK + 1];                /* classes with high bits b: res[off[b] .. off[b+1]) */
    uint64_t  alive[K_MAX + 1], desc[K_MAX + 1], merge[K_MAX + 1], join[K_MAX + 1];
    double    secs;
} sieve2_t;

static void build_sieve2(sieve2_t *out, int nthreads)
{
    double t0 = now_sec();
    memset(out, 0, sizeof *out);
    acc2_t seq;
    memset(&seq, 0, sizeof seq);
    int split = G_K < SPLIT_K ? G_K : SPLIT_K;
    seq.stop = split;
    dfs2(0, 0, 0, 0, 0, &seq);
    acc2_t *accs = NULL;
    int nacc = 0;
    if (split < G_K) {
        W2.fr = seq.fr; W2.nfr = seq.nfr; W2.split = split;
        atomic_store(&W2.next, 0);
        nacc = nthreads;
        accs = calloc((size_t)nacc, sizeof *accs);
        pthread_t *th = calloc((size_t)nacc, sizeof *th);
        if (!accs || !th) { fprintf(stderr, "candidates: out of memory\n"); exit(1); }
        for (int i = 1; i < nacc; i++)
            if (spawn_thread(&th[i], worker2, &accs[i])) { nacc = i; break; }
        worker2(&accs[0]);
        for (int i = 1; i < nacc; i++) pthread_join(th[i], NULL);
        free(th);
        free(seq.fr);
    }
    /* Merges the pieces (from the sequential phase and the threads) bucket by bucket, freeing each. */
    size_t tot = 0, maxb = 0;
    for (int b = 0; b < NBUCK; b++) {
        size_t nb = seq.n[b];
        for (int i = 0; i < nacc; i++) nb += accs[i].n[b];
        out->off[b] = tot;
        tot += nb;
        if (nb > maxb) maxb = nb;
    }
    out->off[NBUCK] = out->n = tot;
    out->res = xrealloc(NULL, tot * sizeof *out->res);
    for (int b = 0; b < NBUCK; b++) {
        size_t at = out->off[b];
        memcpy(out->res + at, seq.v[b], seq.n[b] * sizeof *out->res);
        at += seq.n[b];
        free(seq.v[b]);
        for (int i = 0; i < nacc; i++) {
            memcpy(out->res + at, accs[i].v[b], accs[i].n[b] * sizeof *out->res);
            at += accs[i].n[b];
            free(accs[i].v[b]);
        }
    }
    uint32_t *tmp = xrealloc(NULL, maxb * sizeof *tmp);
    for (int b = 0; b < NBUCK; b++)
        radix_sort32(out->res + out->off[b], out->off[b + 1] - out->off[b], tmp);
    free(tmp);
    for (int k = 0; k <= G_K; k++) {
        out->alive[k] = seq.alive[k]; out->desc[k] = seq.desc[k];
        out->merge[k] = seq.merge[k]; out->join[k] = seq.join[k];
        for (int i = 0; i < nacc; i++) {
            out->alive[k] += accs[i].alive[k]; out->desc[k] += accs[i].desc[k];
            out->merge[k] += accs[i].merge[k]; out->join[k] += accs[i].join[k];
        }
    }
    free(accs);
    out->secs = now_sec() - t0;
}

/* ------------------------------------------------------------------------- */
/* 3^L sieve                                                                 */

/* n ≡ nu (mod 3^L): looks for m < n with T^s(m) = n, using only n mod 3^L. */
static int test3(int L, uint64_t nu, u128 S, witness_t *w)
{
    u128 M = P3[L];
    u128 qmin = S > nu ? (S - nu + M - 1) / M : 0;
    w->k = 0; w->Lw = L;
    if (L == 0) return 0;
    return back_search(M, floor_log2_u128(M), L * LOG2_3, nu, L, 0, 0, nu, qmin, w);
}

static struct {
    uint64_t     *bits;      /* bit nu = 1 ⇔ the class nu mod 3^L survives */
    uint64_t      mod, alive;
    atomic_size_t next;      /* next block of 64·1024 residues */
    double        secs;
} G3;

static void *worker3(void *arg)
{
    uint64_t *alive = arg;
    const uint64_t BLK = 64 * 1024;
    for (;;) {
        uint64_t start = (uint64_t)atomic_fetch_add(&G3.next, 1) * BLK;
        if (start >= G3.mod) break;
        uint64_t end = start + BLK < G3.mod ? start + BLK : G3.mod;
        for (uint64_t nu = start; nu < end; nu++) {
            witness_t w;
            if (!test3(G_L, nu, G_S, &w)) {
                G3.bits[nu >> 6] |= (uint64_t)1 << (nu & 63);   /* blocks are multiples of 64: no contention */
                (*alive)++;
            }
        }
    }
    return NULL;
}

static void build_sieve3(int nthreads)
{
    double t0 = now_sec();
    G3.mod = (uint64_t)P3[G_L];
    G3.bits = calloc((size_t)(G3.mod / 64 + 1), sizeof *G3.bits);
    if (!G3.bits) { fprintf(stderr, "candidates: out of memory\n"); exit(1); }
    atomic_store(&G3.next, 0);
    int nt = G3.mod < 1000000 ? 1 : nthreads;
    uint64_t *alive = calloc((size_t)nt, sizeof *alive);
    pthread_t *th = calloc((size_t)nt, sizeof *th);
    if (!alive || !th) { fprintf(stderr, "candidates: out of memory\n"); exit(1); }
    int started = 1;
    for (int i = 1; i < nt; i++, started++)
        if (spawn_thread(&th[i], worker3, &alive[i])) break;
    worker3(&alive[0]);
    for (int i = 1; i < started; i++) pthread_join(th[i], NULL);
    G3.alive = 0;
    for (int i = 0; i < nt; i++) G3.alive += alive[i];
    free(alive);
    free(th);
    G3.secs = now_sec() - t0;
}

static inline int alive3(uint64_t nu) { return (int)((G3.bits[nu >> 6] >> (nu & 63)) & 1); }

/* ------------------------------------------------------------------------- */
/* Joint table 2^k0 × 3^J                                                    */
/* Knowing also n mod 3^J, the point x_k = T^k(n) becomes known modulo 3^(a+J), which allows
   more odd steps in the backward search. Almost all of this gain appears at shallow levels, so
   a small table suffices: live classes mod 2^k0 (k0 = 16) × residues mod 3^J. It is
   intersected with the 2^K and 3^L tables: n is a candidate only if it survives all three. */

#ifndef JOINT_K
#define JOINT_K 16
#endif
#define J_DEF   9
#define J_MAX   12
static int G_J = J_DEF;              /* -j: 3-adic precision of the joint table (0 = disabled) */

static struct {
    int        k0;
    uint64_t   m3;                   /* 3^J */
    int32_t   *row;                  /* r mod 2^k0 → table row (−1: class killed by the 2^k0 sieve) */
    uint64_t  *bits;                 /* rows of 3^J bits (bit ν = 1 ⇔ live pair) */
    size_t     words, nrows;
    uint64_t   inv2k[K_MAX + 1];     /* 2^(−k) mod 3^J */
    uint64_t   alive, pure_alive;    /* live pairs; pairs the plain 2^k0 × 3^J product would leave */
    double     secs;
} GJ;

static inline uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m) { return (uint64_t)((u128)a * b % m); }

/* Joint class n ≡ r (mod 2^k), n ≡ nu (mod 3^J). Writing n = R + q·M with M = 2^k·3^J,
   x_k = 3^(a+J)·q + Y. Incremental search, as in test2 (descent was already tested there). */
static int test_joint(int k, uint64_t r, int a, u128 y, int last_odd, int J, uint64_t nu, u128 S, witness_t *w)
{
    uint64_t m3 = (uint64_t)P3[J];
    uint64_t q0 = mulmod((nu + m3 - r % m3) % m3, GJ.inv2k[k], m3);   /* q ≡ (ν − r)·2^(−k) */
    u128 M = ((u128)1 << k) * m3, R = r + ((u128)q0 << k), Y = P3[a] * q0 + y;
    u128 qmin = S > R ? (S - R + M - 1) / M : 0;
    int A = a + J, maxe = floor_log2_u128(M);
    double logM = k + J * LOG2_3;
    w->k = k; w->Lw = J;
    if (last_odd)
        return back_search(M, maxe, logM, R, A, 1, 0, 2 * Y, qmin, w);
    if (mod3(Y) == 2)
        return back_search(M, maxe, logM, R, A, 1, 1, div3_exact(2 * Y - 1), qmin, w);
    return 0;
}

typedef struct { size_t w0, w1; uint64_t *stack; uint64_t alive, pure; } jw_t;

/* Walks the 2-adic tree down to k0 carrying the mask of live ν (only the words [w0, w1)). */
static void dfsJ(int k, uint64_t r, int a, u128 y, int last_odd, jw_t *jw, int depth)
{
    size_t nw = jw->w1 - jw->w0;
    uint64_t *mask = jw->stack + (size_t)depth * nw;
    if (k >= 1) {
        witness_t w;
        if (test2(k, r, a, y, last_odd, G_S, &w)) return;
        for (size_t i = 0; i < nw; i++) {
            uint64_t bitsw = mask[i];
            while (bitsw) {
                int b = __builtin_ctzll(bitsw);
                bitsw &= bitsw - 1;
                uint64_t nu = (uint64_t)(jw->w0 + i) * 64 + (uint64_t)b;
                if (test_joint(k, r, a, y, last_odd, G_J, nu, G_S, &w)) mask[i] &= ~((uint64_t)1 << b);
            }
        }
    }
    if (k == GJ.k0) {
        int32_t row = GJ.row[r];
        memcpy(GJ.bits + (size_t)row * GJ.words + jw->w0, mask, nw * sizeof *mask);
        for (size_t i = 0; i < nw; i++) jw->alive += (uint64_t)__builtin_popcountll(mask[i]);
        return;
    }
    uint64_t *child = mask + nw;
    for (int b = 0; b < 2; b++) {
        memcpy(child, mask, nw * sizeof *mask);
        uint64_t rr = r | ((uint64_t)b << k);
        u128 yy = b ? y + P3[a] : y;
        if (yy & 1) dfsJ(k + 1, rr, a + 1, (3 * yy + 1) >> 1, 1, jw, depth + 1);
        else        dfsJ(k + 1, rr, a, yy >> 1, 0, jw, depth + 1);
    }
}

static void *workerJ(void *arg)
{
    jw_t *jw = arg;
    size_t nw = jw->w1 - jw->w0;
    if (!nw) return NULL;
    /* initial mask: the ν that survive the plain 3^J sieve (search from n itself) */
    uint64_t *m0 = jw->stack;
    memset(m0, 0, nw * sizeof *m0);
    for (size_t i = 0; i < nw; i++)
        for (int b = 0; b < 64; b++) {
            uint64_t nu = (uint64_t)(jw->w0 + i) * 64 + (uint64_t)b;
            witness_t w;
            if (nu < GJ.m3 && !test3(G_J, nu, G_S, &w)) m0[i] |= (uint64_t)1 << b;
        }
    for (size_t i = 0; i < nw; i++) jw->pure += (uint64_t)__builtin_popcountll(m0[i]);
    dfsJ(0, 0, 0, 0, 0, jw, 0);
    return NULL;
}

static void rowsJ(int k, uint64_t r, int a, u128 y, int last_odd)
{
    witness_t w;
    if (k >= 1 && test2(k, r, a, y, last_odd, G_S, &w)) return;
    if (k == GJ.k0) { GJ.row[r] = (int32_t)GJ.nrows++; return; }
    for (int b = 0; b < 2; b++) {
        uint64_t rr = r | ((uint64_t)b << k);
        u128 yy = b ? y + P3[a] : y;
        if (yy & 1) rowsJ(k + 1, rr, a + 1, (3 * yy + 1) >> 1, 1);
        else        rowsJ(k + 1, rr, a, yy >> 1, 0);
    }
}

static void init_joint(void)       /* constants used by test_joint (also with -x) */
{
    GJ.k0 = G_K < JOINT_K ? G_K : JOINT_K;
    GJ.m3 = (uint64_t)P3[G_J];
    uint64_t inv2 = (GJ.m3 + 1) / 2;
    GJ.inv2k[0] = 1 % GJ.m3;
    for (int k = 1; k <= K_MAX; k++) GJ.inv2k[k] = mulmod(GJ.inv2k[k - 1], inv2, GJ.m3);
}

static void build_joint(int nthreads)
{
    double t0 = now_sec();
    init_joint();
    GJ.row = malloc(((size_t)1 << GJ.k0) * sizeof *GJ.row);
    if (!GJ.row) { fprintf(stderr, "candidates: out of memory\n"); exit(1); }
    for (size_t i = 0; i < ((size_t)1 << GJ.k0); i++) GJ.row[i] = -1;
    GJ.nrows = 0;
    rowsJ(0, 0, 0, 0, 0);
    GJ.words = (size_t)(GJ.m3 + 63) / 64;
    GJ.bits = calloc(GJ.nrows * GJ.words, sizeof *GJ.bits);
    if (!GJ.bits) { fprintf(stderr, "candidates: out of memory\n"); exit(1); }
    int nt = nthreads;
    if ((size_t)nt > GJ.words) nt = (int)GJ.words;
    jw_t *jw = calloc((size_t)nt, sizeof *jw);
    pthread_t *th = calloc((size_t)nt, sizeof *th);
    if (!jw || !th) { fprintf(stderr, "candidates: out of memory\n"); exit(1); }
    for (int i = 0; i < nt; i++) {
        jw[i].w0 = GJ.words * (size_t)i / (size_t)nt;
        jw[i].w1 = GJ.words * (size_t)(i + 1) / (size_t)nt;
        jw[i].stack = calloc((size_t)(GJ.k0 + 2) * (jw[i].w1 - jw[i].w0 + 1), sizeof *jw[i].stack);
        if (!jw[i].stack) { fprintf(stderr, "candidates: out of memory\n"); exit(1); }
    }
    int started = 1;
    for (int i = 1; i < nt; i++, started++)
        if (spawn_thread(&th[i], workerJ, &jw[i])) break;
    workerJ(&jw[0]);
    for (int i = 1; i < started; i++) pthread_join(th[i], NULL);
    for (int i = started; i < nt; i++) workerJ(&jw[i]);     /* if some thread could not be created */
    GJ.alive = GJ.pure_alive = 0;
    for (int i = 0; i < nt; i++) {
        GJ.alive += jw[i].alive;
        GJ.pure_alive += jw[i].pure;
        free(jw[i].stack);
    }
    GJ.pure_alive *= GJ.nrows;
    free(jw);
    free(th);
    GJ.secs = now_sec() - t0;
}

static inline int aliveJ(uint64_t r, uint64_t nu)
{
    int32_t row = GJ.row[r & (((uint64_t)1 << GJ.k0) - 1)];
    if (row < 0) return 0;
    return (int)((GJ.bits[(size_t)row * GJ.words + (nu >> 6)] >> (nu & 63)) & 1);
}

/* ------------------------------------------------------------------------- */
/* Certificates for a specific n (--explain, --witnesses)                    */

/* Follows n down the 2^K sieve tree, tests n mod 3^L and then the joint table, exactly as the
   tables are built. Returns 1 and the certificate if n is discarded.
   n_mod2K = n mod 2^K, n_mod3L = n mod 3^L, n_mod3J = n mod 3^J. */
static int certify(uint64_t n_mod2K, uint64_t n_mod3L, uint64_t n_mod3J, u128 S, witness_t *w)
{
    uint64_t r = 0;
    int a = 0, last_odd = 0;
    u128 y = 0;
    for (int k = 0; k < G_K; k++) {
        int b = (int)((n_mod2K >> k) & 1);
        u128 yy = b ? y + P3[a] : y;
        r |= (uint64_t)b << k;
        if (yy & 1) { a++; y = (3 * yy + 1) >> 1; last_odd = 1; }
        else        { y = yy >> 1; last_odd = 0; }
        if (test2(k + 1, r, a, y, last_odd, S, w)) return 1;
    }
    if (test3(G_L, n_mod3L, S, w)) return 1;
    if (!G_J) return 0;
    if (test3(G_J, n_mod3J, S, w)) return 1;          /* root of the joint table */
    int k0 = G_K < JOINT_K ? G_K : JOINT_K;
    r = 0; a = 0; y = 0;
    for (int k = 0; k < k0; k++) {
        int b = (int)((n_mod2K >> k) & 1);
        u128 yy = b ? y + P3[a] : y;
        r |= (uint64_t)b << k;
        if (yy & 1) { a++; y = (3 * yy + 1) >> 1; last_odd = 1; }
        else        { y = yy >> 1; last_odd = 0; }
        if (test_joint(k + 1, r, a, y, last_odd, G_J, n_mod3J, S, w)) return 1;
    }
    return 0;
}

static void T_steps_mpz(mpz_t x, uint64_t k)
{
    for (uint64_t i = 0; i < k; i++) {
        if (mpz_odd_p(x)) { mpz_mul_ui(x, x, 3); mpz_add_ui(x, x, 1); }
        mpz_fdiv_q_2exp(x, x, 1);
    }
}

static void mpz_set_u128(mpz_t z, u128 v)
{
    mpz_set_ui(z, (unsigned long)(uint64_t)(v >> 64));
    mpz_mul_2exp(z, z, 64);
    mpz_add_ui(z, z, (unsigned long)(uint64_t)v);
}

/* m = 2^s·3^(A−t)·q + Z, with q = (n − R)/M. Checks 1 ≤ m < n and T^k(n) = T^s(m). */
static int witness_value(const mpz_t n, const witness_t *w, mpz_t m, mpz_t meet)
{
    mpz_t q, tmp;
    mpz_inits(q, tmp, NULL);
    mpz_fdiv_q_2exp(q, n, (mp_bitcnt_t)w->k);              /* q = ⌊n / (2^k·3^Lw)⌋ */
    mpz_ui_pow_ui(tmp, 3, (unsigned long)w->Lw);
    mpz_fdiv_q(q, q, tmp);
    mpz_ui_pow_ui(tmp, 3, (unsigned long)(w->A - w->t));
    mpz_mul_2exp(tmp, tmp, (mp_bitcnt_t)w->s);
    mpz_mul(m, q, tmp);
    mpz_set_u128(tmp, w->Z);
    mpz_add(m, m, tmp);
    int ok = mpz_cmp_ui(m, 1) >= 0 && mpz_cmp(m, n) < 0;
    mpz_set(meet, n);
    T_steps_mpz(meet, (uint64_t)w->k);
    mpz_set(tmp, m);
    T_steps_mpz(tmp, (uint64_t)w->s);
    ok = ok && mpz_cmp(meet, tmp) == 0;
    mpz_clears(q, tmp, NULL);
    return ok;
}

/* ------------------------------------------------------------------------- */
/* Tested: numbers that collatz has already proven not to be counterexamples */
/* Kept in Tested/ (format in tested.h): tested.bin, in increasing order, plus the journal of
   the last run. These numbers are neither generated nor counted again. */

/* Integrity warnings, after reading: anything corrupt is ignored (and may be generated again,
   which is harmless); nothing corrupt excludes a number. */
static void tx_warn(const tx_reader_t *r)
{
    static int warned;
    if (warned || !(r->corrupt || r->bad_blocks || r->bad_recs)) return;
    warned = 1;
    if (r->corrupt)
        fprintf(stderr, "candidates: warning: %s is corrupt and was ignored; its numbers may"
                        " be generated again\n", TX.bin);
    if (r->bad_blocks)
        fprintf(stderr, "candidates: warning: %" PRIu64 " corrupt block(s) in %s were ignored\n",
                r->bad_blocks, TX.bin);
    if (r->bad_recs)
        fprintf(stderr, "candidates: warning: %" PRIu64 " invalid record(s) in the journal or in"
                        " tested.txt were ignored\n", r->bad_recs);
}

static int tx_contains(u128 n)               /* has n already been tested? */
{
    tx_reader_t r;
    tx_u128 v;
    tx_reader_open(&r, n, n + 1);
    int found = tx_reader_next(&r, &v) && v == n;
    tx_warn(&r);
    tx_reader_close(&r);
    return found;
}

/* ------------------------------------------------------------------------- */
/* Number parsing: decimal, 0x, 0b, + - * / % ^ (or **) and parentheses       */

typedef struct { const char *s; size_t i; const char *err; int depth; } parser_t;

static int p_expr(parser_t *p, mpz_t r);

static int p_number(parser_t *p, mpz_t r)
{
    const char *s = p->s + p->i;
    int base = 10;
    if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) { base = 16; s += 2; }
    else if (s[0] == '0' && (s[1] == 'b' || s[1] == 'B')) { base = 2; s += 2; }
    size_t len = 0;
    while (s[len] && (base == 16 ? isxdigit((unsigned char)s[len])
                      : base == 2 ? (s[len] == '0' || s[len] == '1')
                      : isdigit((unsigned char)s[len]))) len++;
    if (!len) { p->err = "number expected"; return 0; }
    char *buf = xrealloc(NULL, len + 1);
    memcpy(buf, s, len);
    buf[len] = 0;
    mpz_set_str(r, buf, base);
    free(buf);
    p->i = (size_t)(s + len - p->s);
    return 1;
}

static int p_atom(parser_t *p, mpz_t r)
{
    if (p->s[p->i] == '(') {
        if (++p->depth > 200) { p->err = "too many parentheses"; return 0; }
        p->i++;
        if (!p_expr(p, r)) return 0;
        if (p->s[p->i] != ')') { p->err = "missing ')'"; return 0; }
        p->i++;
        p->depth--;
        return 1;
    }
    return p_number(p, r);
}

static int p_unary(parser_t *p, mpz_t r);

static int p_power(parser_t *p, mpz_t r)
{
    if (!p_atom(p, r)) return 0;
    if (p->s[p->i] != '^') return 1;
    p->i++;
    if (++p->depth > 200) { p->err = "powers nested too deeply"; return 0; }
    mpz_t e;
    mpz_init(e);
    int ok = p_unary(p, e);
    p->depth--;
    if (ok && mpz_sgn(e) < 0) { p->err = "negative exponent"; ok = 0; }
    if (ok) {
        size_t bits = mpz_sizeinbase(r, 2);
        if (mpz_cmpabs_ui(r, 1) > 0 && (!mpz_fits_ulong_p(e) || (double)mpz_get_ui(e) * (double)bits > 1e8)) {
            p->err = "power too large";
            ok = 0;
        } else if (mpz_cmpabs_ui(r, 1) <= 0) {
            int odd = mpz_odd_p(e);
            if (mpz_sgn(e) == 0) mpz_set_ui(r, 1);
            else if (mpz_cmp_si(r, -1) == 0 && !odd) mpz_set_ui(r, 1);
        } else {
            mpz_pow_ui(r, r, mpz_get_ui(e));
        }
    }
    mpz_clear(e);
    return ok;
}

static int p_unary(parser_t *p, mpz_t r)
{
    int neg = 0;
    while (p->s[p->i] == '+' || p->s[p->i] == '-') neg ^= p->s[p->i++] == '-';
    if (!p_power(p, r)) return 0;
    if (neg) mpz_neg(r, r);
    return 1;
}

static int p_term(parser_t *p, mpz_t r)
{
    if (!p_unary(p, r)) return 0;
    mpz_t b;
    mpz_init(b);
    int ok = 1;
    while (ok && (p->s[p->i] == '*' || p->s[p->i] == '/' || p->s[p->i] == '%')) {
        char op = p->s[p->i++];
        if (!(ok = p_unary(p, b))) break;
        if (op == '*') mpz_mul(r, r, b);
        else if (mpz_sgn(b) == 0) { p->err = "division by zero"; ok = 0; }
        else if (op == '/') mpz_fdiv_q(r, r, b);
        else mpz_fdiv_r(r, r, b);
    }
    mpz_clear(b);
    return ok;
}

static int p_expr(parser_t *p, mpz_t r)
{
    if (!p_term(p, r)) return 0;
    mpz_t b;
    mpz_init(b);
    int ok = 1;
    while (ok && (p->s[p->i] == '+' || p->s[p->i] == '-')) {
        char op = p->s[p->i++];
        if (!(ok = p_term(p, b))) break;
        if (op == '+') mpz_add(r, r, b);
        else mpz_sub(r, r, b);
    }
    mpz_clear(b);
    return ok;
}

/* "1.000.000" (thousands separated by dots, in groups of 3) */
static int dotted_thousands(const char *s)
{
    size_t i = 0;
    while (isdigit((unsigned char)s[i])) i++;
    if (i < 1 || i > 3 || s[i] != '.') return 0;
    while (s[i] == '.') {
        size_t d = 0;
        for (i++; isdigit((unsigned char)s[i]); i++) d++;
        if (d != 3) return 0;
    }
    return s[i] == 0;
}

/* Parses an expression; on error, prints the message and returns 0. */
static int parse_number(const char *text, mpz_t out, const char *what)
{
    size_t len = strlen(text);
    char *norm = xrealloc(NULL, len + 1);
    size_t j = 0;
    for (size_t i = 0; i < len; i++) {
        char c = text[i];
        if (c == ' ' || c == '_' || c == '\t') continue;
        if (c == '*' && text[i + 1] == '*') { norm[j++] = '^'; i++; continue; }
        norm[j++] = c;
    }
    norm[j] = 0;
    if (dotted_thousands(norm)) {
        size_t k = 0;
        for (size_t i = 0; norm[i]; i++)
            if (norm[i] != '.') norm[k++] = norm[i];
        norm[k] = 0;
    }
    parser_t p = { norm, 0, NULL, 0 };
    int ok = j > 0 && p_expr(&p, out);
    if (ok && norm[p.i]) { p.err = "unexpected character"; ok = 0; }
    if (!ok)
        fprintf(stderr, "candidates: invalid %s: \"%s\" (%s)\n", what, text,
                j == 0 ? "empty" : p.err ? p.err : "error");
    free(norm);
    return ok;
}

static int mpz_to_u128(const mpz_t z, u128 *out)
{
    if (mpz_sgn(z) < 0 || mpz_sizeinbase(z, 2) > 128) return 0;
    mpz_t t;
    mpz_init(t);
    mpz_fdiv_q_2exp(t, z, 64);
    u128 v = ((u128)mpz_get_ui(t) << 64);
    mpz_fdiv_r_2exp(t, z, 64);
    v |= (u128)mpz_get_ui(t);
    mpz_clear(t);
    *out = v;
    return 1;
}

/* ------------------------------------------------------------------------- */
/* Output                                                                    */

/* Stop requested by a signal (see cand_on_signal). */
static volatile sig_atomic_t cand_stop = 0, cand_output = 0;
static int out_is_pipe = 0;

/* Numbers are output through a private buffer, always holding complete lines, written with
   write(). When stopping, each write() carries only whole lines (on a pipe, up to PIPE_BUF bytes:
   atomic), so the reader never receives half a line. */
static char  outbuf[1 << 16];
static size_t outlen;

static void cand_die(int sig)                     /* exits through the signal (code 130, 141, ...) */
{
    signal(sig, SIG_DFL);
    raise(sig);
    _exit(128 + sig);
}

static void out_flush(void)
{
    size_t off = 0;
    while (off < outlen) {
        size_t chunk = outlen - off;
        if (cand_stop && out_is_pipe && chunk > PIPE_BUF) {
            chunk = PIPE_BUF;
            while (chunk > 1 && outbuf[off + chunk - 1] != '\n') chunk--;
        }
        ssize_t w = write(STDOUT_FILENO, outbuf + off, chunk);
        if (w < 0) {
            if (errno == EINTR) continue;
            outlen = 0;
            if (errno == EPIPE)                   /* the reader stopped (e.g. | head) */
                cand_die(cand_stop && cand_stop != SIGPIPE ? cand_stop : SIGPIPE);
            fprintf(stderr, "candidates: error writing the output: %s\n", strerror(errno));
            exit(1);
        }
        off += (size_t)w;
    }
    outlen = 0;
}

static void out_str(const char *s, size_t n)      /* n ≤ 4096; the caller passes whole lines */
{
    if (outlen + n > sizeof outbuf) out_flush();
    memcpy(outbuf + outlen, s, n);
    outlen += n;
}

static inline void out_u128(u128 v)
{
    if (outlen > sizeof outbuf - 64) out_flush();
    char tmp[48], *end = tmp + sizeof tmp;
    char *p = u128_dec(end, v);
    size_t n = (size_t)(end - p);
    memcpy(outbuf + outlen, p, n);
    outlen += n;
    outbuf[outlen++] = '\n';
}

/* Stop requested by a signal (Ctrl+C, kill, closed terminal, closed output). While the program
   writes numbers, the 1st signal only records the request: writing stops between two lines, the
   output is delivered up to the last complete line and the program exits through the same
   signal. Outside writing (building tables, counting, drawing), nothing is half done and it
   exits immediately. A 2nd signal exits immediately in any case. */

static void cand_on_signal(int sig)
{
    if (sig == SIGPIPE) {                        /* the reader stopped: subsequent writes fail */
        signal(SIGPIPE, SIG_IGN);
        if (!cand_stop) cand_stop = SIGPIPE;
        return;
    }
    if (cand_output && (!cand_stop || cand_stop == SIGPIPE)) {
        cand_stop = sig;
        return;
    }
    signal(sig, SIG_DFL);
    raise(sig);
}

/* Creates a helper thread with the stop signals blocked: they always go to the main thread. */
static int spawn_thread(pthread_t *th, void *(*fn)(void *), void *arg)
{
    sigset_t block, old;
    sigemptyset(&block);
    sigaddset(&block, SIGINT);
    sigaddset(&block, SIGTERM);
    sigaddset(&block, SIGHUP);
    sigaddset(&block, SIGPIPE);
    pthread_sigmask(SIG_BLOCK, &block, &old);
    int rc = pthread_create(th, NULL, fn, arg);
    pthread_sigmask(SIG_SETMASK, &old, NULL);
    return rc;
}

static void cand_signals_init(void)
{
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = cand_on_signal;
    sa.sa_flags = SA_RESTART;
    sigemptyset(&sa.sa_mask);
    int sigs[] = { SIGINT, SIGTERM, SIGHUP, SIGPIPE };
    for (size_t i = 0; i < sizeof sigs / sizeof *sigs; i++) {
        struct sigaction old;
        if (sigaction(sigs[i], NULL, &old) == 0 && old.sa_handler == SIG_IGN) continue;
        sigaction(sigs[i], &sa, NULL);
    }
}

/* Pseudorandom generator (xoshiro256**, seeded by splitmix64): fast and reproducible with
   --seed. */
static uint64_t RNG[4];

static inline uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static uint64_t rng64(void)
{
    uint64_t r = rotl64(RNG[1] * 5, 7) * 9, t = RNG[1] << 17;
    RNG[2] ^= RNG[0];
    RNG[3] ^= RNG[1];
    RNG[1] ^= RNG[2];
    RNG[0] ^= RNG[3];
    RNG[2] ^= t;
    RNG[3] = rotl64(RNG[3], 45);
    return r;
}

static void rng_seed(uint64_t seed)
{
    uint64_t z = seed;
    for (int i = 0; i < 4; i++) {
        z += 0x9E3779B97F4A7C15ULL;
        uint64_t x = z;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        RNG[i] = x ^ (x >> 31);
    }
}

static uint64_t random_seed(void)
{
    uint64_t s = 0;
#if defined(__APPLE__)
    arc4random_buf(&s, sizeof s);
#else
    int fd = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    if (fd >= 0) { if (read(fd, &s, sizeof s) != (ssize_t)sizeof s) s = 0; close(fd); }
    if (!s) s = (uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32);
#endif
    return s;
}

static u128 rng_below(u128 n)                    /* uniform in [0, n), n ≥ 1, unbiased */
{
    if (n <= 1) return 0;
    int bits = floor_log2_u128(n - 1) + 1;
    u128 mask = bits >= 128 ? ~(u128)0 : ((u128)1 << bits) - 1;
    for (;;) {
        uint64_t hi = rng64(), lo = rng64();      /* in this order: the seed reproduces on any compiler */
        u128 r = ((u128)hi << 64 | lo) & mask;
        if (r < n) return r;
    }
}

/* Reservoir sampling (algorithm R): while walking the candidates, keeps `cap` of them so that,
   at the end, every subset of `cap` candidates is equally likely. A single pass, with a single
   snapshot of the tested numbers (the same one the enumeration uses to skip them). */
static struct { int on; u128 *v; uint64_t cap, n, alloc, seen; } RES;

static void res_add(u128 x)
{
    if (RES.seen < RES.cap) {                     /* still filling (grows gradually: N may be ≫ total) */
        if (RES.n == RES.alloc) {
            RES.alloc = RES.alloc ? 2 * RES.alloc : 4096;
            if (RES.alloc > RES.cap) RES.alloc = RES.cap;
            RES.v = xrealloc(RES.v, RES.alloc * sizeof *RES.v);
        }
        RES.v[RES.n++] = x;
    } else {
        u128 j = rng_below((u128)RES.seen + 1);
        if (j < RES.cap) RES.v[j] = x;
    }
    RES.seen++;
}

static void print_mpz(FILE *f, const mpz_t z)
{
    size_t digits = mpz_sizeinbase(z, 10);
    if (digits <= 80) {
        mpz_out_str(f, 10, z);
        return;
    }
    fprintf(f, "(%zu-bit number)", mpz_sizeinbase(z, 2));
}

/* ------------------------------------------------------------------------- */
/* Modes                                                                     */

static sieve2_t G2;

/* Exact fraction of the integers that survive all three tables (period 2^K·3^max(J,L)). */
static double exact_density(void)
{
    double d2 = (double)G2.n / (double)((uint64_t)1 << G_K);
    double d3 = (double)G3.alive / (double)G3.mod;
    if (!G_J) return d2 * d3;
    /* c[ν_J] = how many ν mod 3^max(J,L), with ν ≡ ν_J (mod 3^J), pass the 3^L sieve */
    uint64_t mJ = GJ.m3, mL = G3.mod, mx = mJ > mL ? mJ : mL;
    double *c = calloc((size_t)mJ, sizeof *c);
    uint64_t *hist = calloc(GJ.nrows ? GJ.nrows : 1, sizeof *hist);
    if (!c || !hist) { fprintf(stderr, "candidates: out of memory\n"); exit(1); }
    for (uint64_t nu = 0; nu < mx; nu++)
        if (alive3(nu % mL)) c[nu % mJ] += 1.0;
    /* how many classes of the 2^K table fall into each row (r mod 2^k0) */
    uint64_t maskk0 = ((uint64_t)1 << GJ.k0) - 1;
    int b = 0;
    for (size_t i = 0; i < G2.n; i++) {
        while (i >= G2.off[b + 1]) b++;
        int32_t row = GJ.row[((uint64_t)b << 32 | G2.res[i]) & maskk0];
        if (row >= 0) hist[row]++;
    }
    double tot = 0;
    for (size_t row = 0; row < GJ.nrows; row++) {
        if (!hist[row]) continue;
        double wsum = 0;
        const uint64_t *bits = GJ.bits + row * GJ.words;
        for (uint64_t nj = 0; nj < mJ; nj++)
            if ((bits[nj >> 6] >> (nj & 63)) & 1) wsum += c[nj];
        tot += wsum * (double)hist[row];
    }
    free(c);
    free(hist);
    return tot / ((double)((uint64_t)1 << G_K) * (double)mx);
}

static void summary(FILE *f)
{
    char a[64], b[32], c[32], d[32], e[32];
    double d2 = (double)G2.n / (double)((uint64_t)1 << G_K);
    double d3 = (double)G3.alive / (double)G3.mod;
    double dt = exact_density();
    fmt_u64(a, G2.n);
    fmt_pct(b, sizeof b, d2, 4);
    fmt_pct(c, sizeof c, d3, 2);
    fmt_pct(d, sizeof d, dt, 4);
    snprintf(e, sizeof e, "%.2f", G2.secs + G3.secs + GJ.secs);
    fprintf(f, "2^%d sieve: %s classes (%s) · 3^%d sieve: %s", G_K, a, b, G_L, c);
    if (G_J) fprintf(f, " · joint table 2^%d × 3^%d", GJ.k0, G_J);
    fprintf(f, " · %s of the integers remain (1 in %.0f) · tables in %s s\n", d, 1.0 / dt, e);
}

static void stats(void)
{
    char a[32], b[32], c[32], d[32], e[32], f[32];
    printf("2^%d sieve (%s; valid for n ≥ ", G_K, G_only_descent ? "descent only" : "descent + trajectory merging");
    char tmp[48];
    tmp[sizeof tmp - 1] = 0;
    fputs(u128_dec(tmp + sizeof tmp - 1, G_S), stdout);
    printf(")\n\n");
    printf("  k         live classes     density     discarded at level k:   descent      merging      joining\n");
    for (int k = 1; k <= G_K; k++) {
        fmt_u64(a, G2.alive[k]);
        fmt_pct(b, sizeof b, (double)G2.alive[k] / (double)((uint64_t)1 << k), 4);
        fmt_u64(c, G2.desc[k]);
        fmt_u64(d, G2.merge[k]);
        fmt_u64(e, G2.join[k]);
        printf(" %2d  %19s   %9s    %32s %12s %12s\n", k, a, b, c, d, e);
    }
    fmt_u64(a, G3.alive);
    fmt_u64(b, G3.mod);
    fmt_pct(c, sizeof c, (double)G3.alive / (double)G3.mod, 4);
    printf("\n3^%d sieve (smaller preimages): %s of %s classes survive (%s)\n", G_L, a, b, c);
    if (G_J) {
        fmt_u64(a, GJ.alive);
        fmt_u64(b, GJ.pure_alive);
        fmt_pct(c, sizeof c, 1.0 - (double)GJ.alive / (double)GJ.pure_alive, 2);
        printf("Joint table 2^%d × 3^%d: %s live (class, residue) pairs, versus %s in the product\n"
               "  of the separate sieves (%s fewer)\n", GJ.k0, G_J, a, b, c);
    }
    double dt = exact_density();
    fmt_pct(d, sizeof d, dt, 5);
    fmt_u64(f, (uint64_t)llround(dt * 1e9));
    snprintf(e, sizeof e, "%.1f", 1.0 / dt);
    printf("Combined: %s of the integers remain (1 in %s), about %s candidates per 10^9 integers.\n",
           d, e, f);
    snprintf(a, sizeof a, "%.2f", G2.secs);
    snprintf(b, sizeof b, "%.2f", G3.secs);
    snprintf(e, sizeof e, "%.2f", GJ.secs);
    printf("Construction: 2^%d sieve in %s s, 3^%d sieve in %s s", G_K, a, G_L, b);
    if (G_J) printf(", joint table in %s s", e);
    printf(".\n");
}

static int explain(const char *text)
{
    mpz_t n, m, meet, two71;
    mpz_inits(n, m, meet, two71, NULL);
    int rc = 0;
    if (!parse_number(text, n, "number")) { rc = 1; goto done; }
    if (mpz_sgn(n) <= 0) {
        fprintf(stderr, "candidates: n must be positive\n");
        rc = 1;
        goto done;
    }
    printf("n = %s", text);
    if (mpz_sizeinbase(n, 10) <= 80 && strspn(text, "0123456789") != strlen(text)) {
        printf(" = ");
        mpz_out_str(stdout, 10, n);
    }
    printf("   [%zu bits]\n", mpz_sizeinbase(n, 2));
    u128 nv;
    int tested = 0;
    if (mpz_sizeinbase(n, 2) > VERIFIED_BITS && mpz_to_u128(n, &nv) && nv < N_LIMIT) {
        tested = tx_contains(nv);
        if (tested)
            printf("✔ ALREADY TESTED: n is in %s, i.e. collatz proved that n is not a\n"
                   "  counterexample. candidates does not generate it again.\n", TX.dir);
    }
    if (mpz_cmp_ui(n, 1) == 0) {
        printf("✔ NOT a counterexample: T(1) = 2 and T(2) = 1.\n");
        goto done;
    }
    mpz_ui_pow_ui(two71, 2, VERIFIED_BITS);
    u128 S;
    if (mpz_cmp(n, two71) < 0) {
        mpz_to_u128(n, &S);
        printf("  (n < 2^71: n has already been verified directly; below, only the sieve analysis)\n");
    } else {
        S = (u128)1 << VERIFIED_BITS;
    }
    uint64_t n2 = (uint64_t)mpz_getlimbn(n, 0) & (((uint64_t)1 << G_K) - 1);
    uint64_t n3 = (uint64_t)mpz_fdiv_ui(n, (unsigned long)P3[G_L]);
    uint64_t nJ = (uint64_t)mpz_fdiv_ui(n, (unsigned long)P3[G_J]);
    witness_t w;
    if (!certify(n2, n3, nJ, S, &w)) {
        if (G_J)
            printf("◯ CANDIDATE: survives the 2^%d and 3^%d sieves and the joint table 2^%d × 3^%d.\n",
                   G_K, G_L, G_K < JOINT_K ? G_K : JOINT_K, G_J);
        else
            printf("◯ CANDIDATE: survives the 2^%d and 3^%d sieves.\n", G_K, G_L);
        printf("  n mod 2^%d = %" PRIu64 ",  n mod 3^%d = %" PRIu64 "\n", G_K, n2, G_L, n3);
        if (tested)
            printf("  No sieve rule discards n, but it has already been tested.\n");
        else
            printf("  No sieve rule discards n; only following the trajectory (./collatz -s)\n"
                   "  can tell whether n is the smallest counterexample.\n");
        goto done;
    }
    int ok = witness_value(n, &w, m, meet);
    mpz_t c, t3;
    mpz_inits(c, t3, NULL);
    printf("✘ DISCARDED: n cannot be the SMALLEST counterexample.\n");
    if (w.k == 0) {
        /* m = (2^s·n − c)/3^t: only n mod 3^t matters (t odd steps backwards). */
        mpz_mul_2exp(c, n, (mp_bitcnt_t)w.s);
        mpz_ui_pow_ui(t3, 3, (unsigned long)w.t);
        mpz_submul(c, t3, m);
        char lab[64];
        if (w.Lw == G_L) snprintf(lab, sizeof lab, "3^%d sieve", G_L);
        else             snprintf(lab, sizeof lab, "joint table (search from n, mod 3^%d)", w.Lw);
        printf("  %s, smaller preimage: n ≡ %" PRIu64 " (mod 3^%d), so\n"
               "  m = (2^%d·n − ", lab, (uint64_t)mpz_fdiv_ui(n, (unsigned long)P3[w.t]), w.t, w.s);
        mpz_out_str(stdout, 10, c);
        printf(")/3^%d is an integer, smaller than n, and T^%d(m) = n.\n", w.t, w.s);
    } else if (w.s == 0) {
        /* T^k(n) = (3^a·n + c)/2^k */
        mpz_mul_2exp(c, meet, (mp_bitcnt_t)w.k);
        mpz_ui_pow_ui(t3, 3, (unsigned long)w.A);
        mpz_submul(c, t3, n);
        printf("  2^%d sieve, descent: n ≡ %" PRIu64 " (mod 2^%d), and for this whole class\n"
               "  T^%d(n) = (3^%d·n + ", G_K, n2 & (((uint64_t)1 << w.k) - 1), w.k, w.k, w.A);
        mpz_out_str(stdout, 10, c);
        printf(")/2^%d < n.\n", w.k);
    } else if (w.Lw) {
        char fix[48];
        if (w.k == 1) snprintf(fix, sizeof fix, "the first step");
        else          snprintf(fix, sizeof fix, "the first %d steps", w.k);
        printf("  joint table: n ≡ %" PRIu64 " (mod 2^%d) fixes %s and, with\n"
               "  n ≡ %" PRIu64 " (mod 3^%d), x = T^%d(n) becomes known modulo 3^%d. This guarantees a\n"
               "  predecessor of x, %d step%s backwards (%d odd), that is smaller than n.\n",
               n2 & (((uint64_t)1 << w.k) - 1), w.k, fix, nJ, w.Lw, w.k, w.A,
               w.s, w.s == 1 ? "" : "s", w.t);
    } else {
        char fix[48];
        if (w.k == 1) snprintf(fix, sizeof fix, "the first step");
        else          snprintf(fix, sizeof fix, "the first %d steps", w.k);
        printf("  2^%d sieve, trajectory merging: n ≡ %" PRIu64 " (mod 2^%d) fixes %s;\n"
               "  x = T^%d(n) is known modulo 3^%d, which guarantees a predecessor of x, %d step%s\n"
               "  backwards (%d odd), that is smaller than n.\n",
               G_K, n2 & (((uint64_t)1 << w.k) - 1), w.k, fix, w.k, w.A, w.s, w.s == 1 ? "" : "s",
               w.t);
    }
    printf("  m = ");
    print_mpz(stdout, m);
    printf("\n  T^%d(n) = T^%d(m) = ", w.k, w.s);
    print_mpz(stdout, meet);
    printf("   (steps of T)\n");
    if (ok) {
        printf("  Checked by direct simulation: 1 ≤ m < n and the trajectories meet.\n"
               "  If n were a counterexample, so would be m, and m < n.\n");
    } else {
        printf("  INTERNAL ERROR: the certificate does not check out. Please report it.\n");
        rc = 4;
    }
    mpz_clears(c, t3, NULL);
done:
    mpz_clears(n, m, meet, two71, NULL);
    return rc;
}

/* --witnesses: one line per n in [A, B): "n C" (candidate) or "n E i j m" (discarded, with
   T^i(n) = T^j(m) and m < n). Lets the tests check every discard by brute force. */
static int witnesses(u128 A, u128 B)
{
    mpz_t n, m, meet;
    mpz_inits(n, m, meet, NULL);
    int rc = 0;
    for (u128 x = A; x < B && !cand_stop; x++) {
        char line[160], tmp[48], *end = tmp + sizeof tmp;
        char *d = u128_dec(end, x);
        size_t len = (size_t)(end - d);
        memcpy(line, d, len);
        witness_t w;
        if (!certify((uint64_t)x & (((uint64_t)1 << G_K) - 1), (uint64_t)(x % P3[G_L]),
                     (uint64_t)(x % P3[G_J]), G_S, &w)) {
            memcpy(line + len, " C\n", 3);
            out_str(line, len + 3);
            continue;
        }
        mpz_set_u128(n, x);
        if (!witness_value(n, &w, m, meet)) rc = 4;
        len += (size_t)snprintf(line + len, sizeof line - len, " E %d %d ", w.k, w.s);
        if (mpz_sizeinbase(m, 10) + len + 2 < sizeof line) {
            mpz_get_str(line + len, 10, m);
            len += strlen(line + len);
        }
        line[len++] = '\n';
        out_str(line, len);
    }
    mpz_clears(n, m, meet, NULL);
    return rc;
}

/* Generates (or counts) the candidates in [A, B), stopping after maxc candidates. */
static struct {
    u128 A, B;
    u128 q0, q1;              /* blocks of 2^K: q = n >> K (up to 2^126) */
    atomic_uint_fast64_t nextq;   /* next block to count: q0 + nextq */
    uint64_t m3, two_k_mod3;  /* 3^L and 2^K mod 3^L */
    uint64_t mJ, two_k_modJ;  /* 3^J and 2^K mod 3^J (joint table) */
} GG;

/* Does n = q·2^K + r survive the 3^L and joint tables? base3 = q·2^K mod 3^L, baseJ likewise mod 3^J. */
static inline int keep(uint64_t r, uint64_t base3, uint64_t baseJ)
{
    uint64_t nu = base3 + r % GG.m3;
    if (nu >= GG.m3) nu -= GG.m3;
    if (!alive3(nu)) return 0;
    if (!G_J) return 1;
    uint64_t nj = baseJ + r % GG.mJ;
    if (nj >= GG.mJ) nj -= GG.mJ;
    return aliveJ(r, nj);
}

/* Index of the 1st class ≥ key in the table (0 ≤ key ≤ 2^K). */
static size_t lower_bound(uint64_t key)
{
    int b = (int)(key >> 32);
    if (b >= NBUCK) return G2.n;
    uint32_t k32 = (uint32_t)key;
    size_t lo = G2.off[b], hi = G2.off[b + 1];
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (G2.res[mid] < k32) lo = mid + 1; else hi = mid;
    }
    return lo;
}

static inline int bucket_of(size_t i)
{
    int b = 0;
    while (b < NBUCK - 1 && i >= G2.off[b + 1]) b++;
    return b;
}

/* Range of table indices for block q. */
static void block_range(u128 q, size_t *i0, size_t *i1)
{
    uint64_t mask = ((uint64_t)1 << G_K) - 1;
    *i0 = q == GG.q0 ? lower_bound((uint64_t)GG.A & mask) : 0;
    *i1 = q == GG.q1 ? lower_bound((uint64_t)((GG.B - 1) & mask) + 1) : G2.n;
}

static uint64_t count_block(u128 q)
{
    size_t i0, i1;
    block_range(q, &i0, &i1);
    uint64_t base3 = mulmod((uint64_t)(q % GG.m3), GG.two_k_mod3, GG.m3), c = 0;
    uint64_t baseJ = mulmod((uint64_t)(q % GG.mJ), GG.two_k_modJ, GG.mJ);
    int b = bucket_of(i0);
    for (size_t i = i0; i < i1; i++) {
        while (i >= G2.off[b + 1]) b++;
        c += (uint64_t)keep((uint64_t)b << 32 | G2.res[i], base3, baseJ);
    }
    return c;
}

static void *count_worker(void *arg)
{
    uint64_t *c = arg;
    for (;;) {
        uint64_t i = atomic_fetch_add(&GG.nextq, 1);
        if (i > GG.q1 - GG.q0) break;
        *c += count_block(GG.q0 + i);
    }
    return NULL;
}

static void setup_range(u128 A, u128 B)
{
    GG.A = A; GG.B = B;
    GG.q0 = A >> G_K;
    GG.q1 = (B - 1) >> G_K;
    GG.m3 = (uint64_t)P3[G_L];
    GG.two_k_mod3 = (uint64_t)(((u128)1 << G_K) % P3[G_L]);
    GG.mJ = (uint64_t)P3[G_J];
    GG.two_k_modJ = (uint64_t)(((u128)1 << G_K) % P3[G_J]);
}

/* Does x survive the sieves? (same tables as generation; x within the configured interval) */
static int is_candidate(u128 x)
{
    uint64_t r = (uint64_t)x & (((uint64_t)1 << G_K) - 1);
    size_t i = lower_bound(r);
    int b = (int)(r >> 32);
    if (i >= G2.off[b + 1] || G2.res[i] != (uint32_t)r) return 0;
    u128 q = x >> G_K;
    uint64_t base3 = mulmod((uint64_t)(q % GG.m3), GG.two_k_mod3, GG.m3);
    uint64_t baseJ = mulmod((uint64_t)(q % GG.mJ), GG.two_k_modJ, GG.mJ);
    return keep(r, base3, baseJ);
}

static uint64_t count_range(u128 A, u128 B, int nthreads)
{
    if (A >= B) return 0;
    setup_range(A, B);
    atomic_store(&GG.nextq, 0);
    uint64_t *c = calloc((size_t)nthreads, sizeof *c);
    pthread_t *th = calloc((size_t)nthreads, sizeof *th);
    if (!c || !th) { fprintf(stderr, "candidates: out of memory\n"); exit(1); }
    int nt = GG.q1 - GG.q0 < 4 ? 1 : nthreads, started = 1;
    if (GG.q1 - GG.q0 >= ((u128)1 << 62)) {
        fprintf(stderr, "candidates: interval too large for --count\n");
        exit(1);
    }
    for (int i = 1; i < nt; i++, started++)
        if (spawn_thread(&th[i], count_worker, &c[i])) break;
    count_worker(&c[0]);
    uint64_t tot = c[0];
    for (int i = 1; i < started; i++) { pthread_join(th[i], NULL); tot += c[i]; }
    free(c);
    free(th);
    tx_reader_t r;                               /* tested numbers of [A, B) do not count */
    tx_u128 v;
    tx_reader_open(&r, A, B);
    while (tx_reader_next(&r, &v)) tot -= (uint64_t)is_candidate(v);
    tx_warn(&r);
    tx_reader_close(&r);
    return tot;
}

/* --tested: lists (or counts) the tested numbers of [A, B), in increasing order. */
static uint64_t list_tested(u128 A, u128 B, int count_only, uint64_t maxc)
{
    tx_reader_t r;
    tx_u128 v;
    uint64_t c = 0;
    tx_reader_open(&r, A, B);
    while (c < maxc && !(cand_stop && !count_only) && tx_reader_next(&r, &v)) {
        c++;
        if (!count_only) out_u128(v);
    }
    tx_warn(&r);
    tx_reader_close(&r);
    return c;
}

static uint64_t print_range(u128 A, u128 B, uint64_t maxc)
{
    if (A >= B || !maxc) return 0;
    setup_range(A, B);
    uint64_t printed = 0;
    tx_reader_t tr;                              /* tested numbers of [A, B), in order, to skip */
    tx_u128 tv = 0;
    tx_reader_open(&tr, A, B);
    int th = tx_reader_next(&tr, &tv);
    for (u128 q = GG.q0;; q++) {
        if (cand_stop) goto done;
        size_t i0, i1;
        block_range(q, &i0, &i1);
        uint64_t base3 = mulmod((uint64_t)(q % GG.m3), GG.two_k_mod3, GG.m3);
        uint64_t baseJ = mulmod((uint64_t)(q % GG.mJ), GG.two_k_modJ, GG.mJ);
        u128 hi = (u128)q << G_K;
        int b = bucket_of(i0);
        for (size_t i = i0; i < i1; i++) {
            while (i >= G2.off[b + 1]) b++;
            uint64_t r = (uint64_t)b << 32 | G2.res[i];
            if (!keep(r, base3, baseJ)) continue;
            u128 n = hi | r;
            while (th && tv < n) th = tx_reader_next(&tr, &tv);
            if (th && tv == n) continue;             /* already tested: not generated again */
            if (RES.on) {                            /* exact sampling: goes to the reservoir */
                res_add(n);
                continue;
            }
            if (cand_stop) goto done;                /* stop requested: end at the last complete line */
            out_u128(n);
            if (++printed == maxc) goto done;
        }
        if (q == GG.q1) break;
    }
done:
    tx_warn(&tr);
    tx_reader_close(&tr);
    return printed;
}

/* ------------------------------------------------------------------------- */
/* Random sampling (-a N): N distinct candidates, uniform among the untested ones of [A, B) */

/* The classes of the 2^K table that fall in [A, B), numbered 0..total−1: the rest of the 1st
   block, the whole blocks in between and the beginning of the last one. Drawing a uniform number
   in this range yields a uniform surviving 2-adic class, without scanning the interval. */
typedef struct { u128 total, nfull; size_t a0, a1, b0, b1; } wnd_t;

static void wnd_init(wnd_t *W)
{
    memset(W, 0, sizeof *W);
    block_range(GG.q0, &W->a0, &W->a1);
    if (GG.q0 == GG.q1) {
        W->total = W->a1 - W->a0;
        return;
    }
    block_range(GG.q1, &W->b0, &W->b1);
    W->nfull = GG.q1 - GG.q0 - 1;
    W->total = (u128)(W->a1 - W->a0) + W->nfull * G2.n + (W->b1 - W->b0);
}

static u128 wnd_number(const wnd_t *W, u128 t, uint64_t *rp, u128 *qp)   /* the t-th class → n */
{
    u128 q, n0 = W->a1 - W->a0, nf = W->nfull * G2.n;
    size_t i;
    if (t < n0) { q = GG.q0; i = W->a0 + (size_t)t; }
    else if ((t -= n0) < nf) { q = GG.q0 + 1 + t / G2.n; i = (size_t)(t % G2.n); }
    else { q = GG.q1; i = W->b0 + (size_t)(t - nf); }
    *rp = (uint64_t)bucket_of(i) << 32 | G2.res[i];
    *qp = q;
    return q << G_K | *rp;
}

/* A uniform candidate of [A, B) (tested numbers not yet considered): draws the 2-adic class and
   accepts it if it passes the 3^L and joint sieves. 0 if stopped or nothing was found. */
static u128 draw_candidate(const wnd_t *W)
{
    for (uint64_t tries = 1; tries < ((uint64_t)1 << 26); tries++) {
        if (!(tries & 1023) && cand_stop) return 0;
        uint64_t r;
        u128 q, n = wnd_number(W, rng_below(W->total), &r, &q);
        uint64_t b3 = mulmod((uint64_t)(q % GG.m3), GG.two_k_mod3, GG.m3);
        uint64_t bJ = mulmod((uint64_t)(q % GG.mJ), GG.two_k_modJ, GG.mJ);
        if (keep(r, b3, bJ)) return n;
    }
    return 0;
}

static int cmp_u128v(const void *a, const void *b)
{
    u128 x = *(const u128 *)a, y = *(const u128 *)b;
    return x < y ? -1 : x > y;
}

/* Removes the already tested numbers from the (sorted) array. */
static size_t drop_tested(u128 *v, size_t n, u128 A, u128 B)
{
    tx_reader_t r;
    tx_u128 t;
    tx_reader_open(&r, A, B);
    int ht = tx_reader_next(&r, &t);
    size_t k = 0;
    for (size_t i = 0; i < n; i++) {
        while (ht && t < v[i]) ht = tx_reader_next(&r, &t);
        if (ht && t == v[i]) continue;
        v[k++] = v[i];
    }
    tx_warn(&r);
    tx_reader_close(&r);
    return k;
}

static size_t sort_unique(u128 *v, size_t n)
{
    if (n < 2) return n;
    qsort(v, n, sizeof *v, cmp_u128v);
    size_t k = 1;
    for (size_t i = 1; i < n; i++)
        if (v[i] != v[k - 1]) v[k++] = v[i];
    return k;
}

#define SAMPLE_MAX ((uint64_t)1 << 27)   /* the drawn numbers stay in memory (16 bytes each) */

/* Prints the drawn numbers (in order), with safe stopping enabled. */
static void sample_output(const u128 *v, size_t n)
{
    cand_output = 1;
    for (size_t i = 0; i < n && !cand_stop; i++) out_u128(v[i]);
}

/* Exact mode: one pass over the candidates of [A, B), with reservoir sampling. */
static int sample_reservoir(u128 A, u128 B, uint64_t N, int verbose)
{
    if (verbose) fprintf(stderr, "random: reservoir sampling (one pass over the candidates)\n");
    memset(&RES, 0, sizeof RES);
    RES.on = 1;
    RES.cap = N;
    print_range(A, B, UINT64_MAX);
    RES.on = 0;
    if (RES.seen < N) {
        char a[32];
        fmt_u64(a, RES.seen);
        if (RES.seen)
            fprintf(stderr, "candidates: warning: [A, B) has only %s untested candidate%s; printing all\n",
                    a, RES.seen == 1 ? "" : "s");
        else
            fprintf(stderr, "candidates: warning: there are no untested candidates in [A, B)\n");
    } else if (verbose) {
        char a[32];
        fmt_u64(a, RES.seen);
        fprintf(stderr, "random: %s untested candidates in the interval\n", a);
    }
    size_t n = sort_unique(RES.v, RES.n);
    sample_output(RES.v, n);
    free(RES.v);
    memset(&RES, 0, sizeof RES);
    return 0;
}

/* Rejection mode (N small compared with the total): draws uniform candidates in batches; each
   batch drops repeats and tested numbers. Together, independent uniform draws form a uniform
   subset; if there are too many, N of them are kept, also at random (algorithm S). Returns 2 if
   the yield is too low: the exact mode then takes over. */
static int sample_rejection(const wnd_t *W, u128 A, u128 B, uint64_t N, double f, int verbose, double est)
{
    if (verbose) fprintf(stderr, "random: rejection sampling (~%.3g candidates in the interval)\n", est);
    u128 *pool = NULL, *batch = NULL;
    size_t np = 0, cap = 0;
    int stuck = 0;
    while (np < N) {
        uint64_t need = N - np, draws = (uint64_t)((double)need / f * 1.25) + 64;
        batch = xrealloc(batch, draws * sizeof *batch);
        size_t nb = 0;
        for (uint64_t d = 0; d < draws; d++) {
            u128 x = draw_candidate(W);
            if (!x) break;
            batch[nb++] = x;
        }
        nb = sort_unique(batch, nb);
        nb = drop_tested(batch, nb, A, B);
        if (np + nb > cap) {
            cap = (np + nb) * 2;
            pool = xrealloc(pool, cap * sizeof *pool);
        }
        size_t before = np;
        memcpy(pool + np, batch, nb * sizeof *batch);
        np = sort_unique(pool, np + nb);
        double got = (double)(np - before);
        f = got / (double)draws;
        if (np > before && f > 0.001) stuck = 0;
        else if (++stuck >= 3) {                  /* low yield: the exact mode handles it */
            free(pool);
            free(batch);
            return 2;
        }
        if (f < 0.001) f = 0.001;
    }
    free(batch);
    uint64_t need = N, left = np;
    size_t k = 0;
    for (size_t i = 0; i < np && need; i++, left--)
        if (rng_below(left) < need) { pool[k++] = pool[i]; need--; }
    sample_output(pool, k);
    free(pool);
    return 0;
}

static int sample_range(u128 A, u128 B, uint64_t N, int verbose, uint64_t seed, int seed_given)
{
    if (!seed_given || verbose)
        fprintf(stderr, "random: seed %" PRIu64 " (use --seed %" PRIu64 " to repeat)\n", seed, seed);
    if (A >= B) {
        fprintf(stderr, "candidates: warning: the interval [A, B) is empty\n");
        return 0;
    }
    setup_range(A, B);
    wnd_t W;
    wnd_init(&W);
    if (W.total == 0) {
        fprintf(stderr, "candidates: warning: there are no candidates in [A, B)\n");
        return 0;
    }
    double d2 = (double)G2.n / (double)((uint64_t)1 << G_K);
    double est = (double)W.total * (exact_density() / d2);   /* expected candidates in the interval */
    double nt = (double)list_tested(A, B, 1, UINT64_MAX);
    /* Rejection when N is small compared with the free candidates and the interval is not
       heavily tested; otherwise (or if rejection yields too little), the exact pass. */
    if ((double)N * 16 < est - nt && nt * 4 < est) {
        if (N > SAMPLE_MAX) {
            fprintf(stderr, "candidates: -a: at most %" PRIu64 " numbers per draw\n", SAMPLE_MAX);
            return 1;
        }
        int rc = sample_rejection(&W, A, B, N, 1.0 - nt / est, verbose, est);
        if (rc != 2) return rc;
        if (verbose) fprintf(stderr, "random: rejection yielded too little; switching to the exact mode\n");
    }
    if (N > SAMPLE_MAX && (double)N < est - nt) {
        fprintf(stderr, "candidates: -a: at most %" PRIu64 " numbers per draw\n", SAMPLE_MAX);
        return 1;
    }
    return sample_reservoir(A, B, N, verbose);
}

/* ------------------------------------------------------------------------- */
/* Command line                                                              */

/* Closes standard output; a write error (disk full, closed output) becomes exit code 1. */
static int finish_output(int rc)
{
    out_flush();
    if (fflush(stdout) == EOF || ferror(stdout)) {
        if (errno == EPIPE) return rc;           /* the reader stopped (e.g. | head): not an error */
        fprintf(stderr, "candidates: error writing the output: %s\n", strerror(errno));
        return rc > 1 ? rc : 1;
    }
    return rc;
}

/* End with output: delivers up to the last complete line; if interrupted, exits through the signal. */
static int cand_finish(int rc)
{
    rc = finish_output(rc);
    if (cand_stop) {
        if (cand_stop != SIGPIPE)
            fprintf(stderr, "\ncandidates: interrupted; the output ends at the last complete line.\n");
        cand_die(cand_stop);
    }
    return rc;
}

static void usage(FILE *f)
{
    fprintf(f,
"Usage: candidates [options] [A [B]]\n"
"Lists, in increasing order, the n of the interval [A, B) that can still be the\n"
"SMALLEST counterexample to the Collatz conjecture. The others are discarded by\n"
"sieves, each with a certificate: an m < n whose trajectory meets that of n.\n"
"Numbers already proven by collatz (Tested/, next to the program, or the directory\n"
"$COLLATZ_TESTED) are not listed either; --tested shows which they are.\n"
"\n"
"  A                    start (default 2^71: every smaller integer is already verified)\n"
"  B                    end, exclusive (default: no end; use -q or | head)\n"
"                       A and B accept expressions: 2^71+10^9, 0x..., 1_000_000\n"
"  -q, --quantity N     stop after N candidates\n"
"  -c, --count          only count the candidates of [A, B) (requires B)\n"
"  -a, --random N       draw N distinct candidates of [A, B) (requires B), uniform\n"
"                       among the untested ones; printed in increasing order\n"
"      --seed S         seed of the draw, to repeat it (without it, the seed used\n"
"                       is printed on standard error)\n"
"  -k, --bits K         2^K sieve (1 to %d, default %d). A larger K discards more but\n"
"                       costs memory and time: 2^32 ~0.3 GB and 1 s; 2^34 ~0.9 GB\n"
"                       and 2-4 s; 2^36 ~2 GB and 8-15 s (Apple M5)\n"
"  -l, --three L        3^L preimage sieve (0 to %d, default %d)\n"
"  -j, --joint J        joint table 2^min(K,16) × 3^J (0 to %d, default %d; 0 disables):\n"
"                       also uses n mod 3^J in trajectory merging\n"
"  -x, --explain n      tell why n was discarded (and show the m), or that it is a\n"
"                       candidate. Accepts several -x and numbers of any size\n"
"  -e, --stats          sieve efficiency, level by level\n"
"  -v, --verbose        sieve summary on standard error\n"
"  -t, --threads T      threads for construction and counting (default: all cores)\n"
"      --descent-only   classical 2^K sieve, descent only (for comparison)\n"
"      --tested         list, in increasing order, the numbers of [A, B) already\n"
"                       tested (Tested/tested.bin); with -c, only count them\n"
"      --witnesses      for each n of [A, B): \"n C\" (candidate) or \"n E i j m\"\n"
"                       (discarded: T^i(n) = T^j(m), m < n). For tests\n"
"  -h, --help           show this help\n"
"\n"
"Example:  ./candidates '2^71' '2^71+10^9' | ./collatz -r -s\n"
"(Quote the expressions: in zsh, an unquoted * becomes a filename pattern.)\n"
"If no line of the collatz output is CYCLE (and none stays unresolved), every\n"
"n < 2^71+10^9 converges. With A > 2^71, the conclusion requires [2^71, A) to have\n"
"been covered first.\n"
"Exit code: 0 success; 1 input error; 4 certificate that did not check out (bug).\n",
            K_MAX, K_DEF, L_MAX, L_DEF, J_MAX, J_DEF);
}

static int parse_int_opt(const char *s, int lo, int hi, int *out, const char *name)
{
    char *end;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || *end || end == s || v < lo || v > hi) {
        fprintf(stderr, "candidates: invalid value for %s: \"%s\" (use %d to %d)\n", name, s, lo, hi);
        return 0;
    }
    *out = (int)v;
    return 1;
}

int main(int argc, char **argv)
{
    init_tables();
    init_gtab();
    tx_init(argv[0]);
    cand_signals_init();
    struct stat ost;
    out_is_pipe = fstat(STDOUT_FILENO, &ost) == 0 && (S_ISFIFO(ost.st_mode) || S_ISSOCK(ost.st_mode));
    const char *pos[2] = { NULL, NULL };
    int npos = 0, opt_count = 0, opt_stats = 0, opt_verbose = 0, opt_wit = 0, opt_tested = 0;
    int nthreads = count_cpus(), have_seed = 0;
    const char *sample_arg = NULL;
    uint64_t seed = 0;
    const char *qty = NULL;
    const char **explain_list = calloc((size_t)argc + 1, sizeof *explain_list);
    int nexplain = 0;
    if (!explain_list) return 1;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *val = NULL;
#define TAKE(name) do { if (i + 1 >= argc) { fprintf(stderr, "candidates: %s requires a value\n", name); return 1; } val = argv[++i]; } while (0)
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(stdout); return 0; }
        else if (!strcmp(a, "-c") || !strcmp(a, "--count")) opt_count = 1;
        else if (!strcmp(a, "-e") || !strcmp(a, "--stats")) opt_stats = 1;
        else if (!strcmp(a, "-v") || !strcmp(a, "--verbose")) opt_verbose = 1;
        else if (!strcmp(a, "--descent-only")) G_only_descent = 1;
        else if (!strcmp(a, "--witnesses")) opt_wit = 1;
        else if (!strcmp(a, "--tested")) opt_tested = 1;
        else if (!strcmp(a, "-a") || !strcmp(a, "--random")) { TAKE(a); sample_arg = val; }
        else if (!strncmp(a, "--random=", 9)) sample_arg = a + 9;
        else if (!strcmp(a, "--seed") || !strncmp(a, "--seed=", 7)) {
            if (a[6] == '=') val = a + 7; else TAKE(a);
            char *end;
            errno = 0;
            unsigned long long v = strtoull(val, &end, 10);
            if (errno || *end || end == val || !isdigit((unsigned char)val[0])) {
                fprintf(stderr, "candidates: invalid seed: \"%s\" (use an integer from 0 to 2^64−1)\n", val);
                return 1;
            }
            seed = v;
            have_seed = 1;
        }
        else if (!strcmp(a, "--no-prune")) G_no_prune = 1;
        else if (!strcmp(a, "-q") || !strcmp(a, "--quantity")) { TAKE(a); qty = val; }
        else if (!strncmp(a, "--quantity=", 11)) qty = a + 11;
        else if (!strcmp(a, "-k") || !strcmp(a, "--bits")) { TAKE(a); if (!parse_int_opt(val, 1, K_MAX, &G_K, "-k")) return 1; }
        else if (!strncmp(a, "--bits=", 7)) { if (!parse_int_opt(a + 7, 1, K_MAX, &G_K, "-k")) return 1; }
        else if (!strcmp(a, "-l") || !strcmp(a, "--three")) { TAKE(a); if (!parse_int_opt(val, 0, L_MAX, &G_L, "-l")) return 1; }
        else if (!strcmp(a, "-j") || !strcmp(a, "--joint")) { TAKE(a); if (!parse_int_opt(val, 0, J_MAX, &G_J, "-j")) return 1; }
        else if (!strncmp(a, "--joint=", 8)) { if (!parse_int_opt(a + 8, 0, J_MAX, &G_J, "-j")) return 1; }
        else if (!strncmp(a, "--three=", 8)) { if (!parse_int_opt(a + 8, 0, L_MAX, &G_L, "-l")) return 1; }
        else if (!strcmp(a, "-t") || !strcmp(a, "--threads")) { TAKE(a); if (!parse_int_opt(val, 1, 64, &nthreads, "-t")) return 1; }
        else if (!strncmp(a, "--threads=", 10)) { if (!parse_int_opt(a + 10, 1, 64, &nthreads, "-t")) return 1; }
        else if (!strcmp(a, "-x") || !strcmp(a, "--explain")) { TAKE(a); explain_list[nexplain++] = val; }
        else if (!strncmp(a, "--explain=", 10)) explain_list[nexplain++] = a + 10;
        else if (a[0] == '-' && a[1] && !isdigit((unsigned char)a[1]) && a[1] != '(') {
            fprintf(stderr, "candidates: unknown option: %s\nUse \"candidates -h\" to see the options.\n", a);
            return 1;
        } else if (npos < 2) pos[npos++] = a;
        else {
            fprintf(stderr, "candidates: too many arguments (the interval is just A and B)\n");
            return 1;
        }
#undef TAKE
    }

    /* Interval */
    u128 A = (u128)1 << VERIFIED_BITS, B = N_LIMIT;
    int haveB = 0;
    mpz_t z;
    mpz_init(z);
    for (int i = 0; i < npos; i++) {
        if (!parse_number(pos[i], z, i == 0 ? "start A" : "end B")) return 1;
        u128 v;
        if (!mpz_to_u128(z, &v) || v > N_LIMIT || (i == 0 && v == N_LIMIT)) {
            fprintf(stderr, "candidates: %s out of range (use %s)\n", i == 0 ? "A" : "B",
                    i == 0 ? "0 ≤ A < 2^127" : "0 ≤ B ≤ 2^127");
            return 1;
        }
        if (i == 0) A = v; else { B = v; haveB = 1; }
    }
    uint64_t maxc = UINT64_MAX;
    if (qty) {
        if (!parse_number(qty, z, "quantity")) return 1;
        if (mpz_sgn(z) < 0 || !mpz_fits_ulong_p(z)) {
            fprintf(stderr, "candidates: invalid quantity\n");
            return 1;
        }
        maxc = mpz_get_ui(z);
    }
    uint64_t nsample = 0;
    if (sample_arg) {
        if (!parse_number(sample_arg, z, "number to draw")) return 1;
        if (mpz_sgn(z) <= 0) {
            fprintf(stderr, "candidates: -a requires a positive quantity\n");
            return 1;
        }
        if (!mpz_fits_ulong_p(z)) {
            fprintf(stderr, "candidates: -a: quantity too large\n");
            return 1;
        }
        nsample = mpz_get_ui(z);
        if (!haveB) { fprintf(stderr, "candidates: -a requires the interval [A, B): e.g. candidates -a 1000 '2^71' '2^80'\n"); return 1; }
        if (opt_count || qty || opt_tested || opt_wit || opt_stats || nexplain) {
            fprintf(stderr, "candidates: -a cannot be combined with -c, -q, -e, -x, --tested or --witnesses\n");
            return 1;
        }
    } else if (have_seed) {
        fprintf(stderr, "candidates: warning: --seed only applies together with -a\n");
    }
    if (!have_seed) seed = random_seed();
    rng_seed(seed);
    mpz_clear(z);
    if (haveB && A > B) {
        fprintf(stderr, "candidates: the start A is greater than the end B\n");
        return 1;
    }
    if (A < 2) A = 2;         /* 1 is not a counterexample: T(1) = 2, T(2) = 1 */
    if (B < A) B = A;         /* [1, 2) and [0, 1): nothing to list */
    if (opt_count && !haveB && !opt_tested) {
        fprintf(stderr, "candidates: --count requires the end B of the interval\n");
        return 1;
    }

    /* The sieve is valid for n ≥ S. Above 2^71, S = 2^71 (the table does not depend on A). */
    G_S = A < ((u128)1 << VERIFIED_BITS) ? A : (u128)1 << VERIFIED_BITS;

    /* --explain does not need the tables: it follows only the path of n. */
    if (G_J) init_joint();
    if (nexplain) {
        int rc = 0;
        setvbuf(stdout, NULL, _IOLBF, 0);          /* explanation: one whole line at a time */
        cand_output = 1;
        for (int i = 0; i < nexplain && !cand_stop; i++) {
            if (i) putchar('\n');
            int r = explain(explain_list[i]);
            if (r > rc) rc = r;
        }
        free(explain_list);
        return finish_output(rc);
    }
    free(explain_list);
    if (opt_tested) {                            /* list or count what is in Tested/ */
        cand_output = !opt_count;
        uint64_t c = list_tested(A, B, opt_count, maxc);
        if (opt_count) {
            char tmp[48];
            tmp[sizeof tmp - 1] = 0;
            puts(u128_dec(tmp + sizeof tmp - 1, c));
        }
        return cand_finish(0);
    }
    if (opt_wit) {
        if (!haveB) { fprintf(stderr, "candidates: --witnesses requires the end B\n"); return 1; }
        if (B > A && B - A > ((u128)1 << 24)) { fprintf(stderr, "candidates: --witnesses: interval too large (max. 2^24)\n"); return 1; }
        cand_output = 1;
        return cand_finish(witnesses(A, B));
    }

    if (G_K > 34 && !opt_stats)
        fprintf(stderr, "candidates: warning: the 2^%d sieve takes a lot of memory and construction time.\n", G_K);
    build_sieve2(&G2, nthreads);
    build_sieve3(nthreads);
    if (G_J) build_joint(nthreads);

    if (opt_stats) {
        stats();
        return 0;
    }
    if (opt_verbose) {
        summary(stderr);
        char a[32];
        uint64_t nt = list_tested(A, B, 1, UINT64_MAX);
        fmt_u64(a, nt);
        if (nt == 1)
            fprintf(stderr, "tested: 1 number of %s is in [A, B) and will not be generated\n", TX.dir);
        else
            fprintf(stderr, "tested: %s numbers of %s are in [A, B) and will not be generated\n", a, TX.dir);
    }

    if (sample_arg)                              /* -a N: random sampling (output only at the end) */
        return cand_finish(sample_range(A, B, nsample, opt_verbose, seed, have_seed));
    if (opt_count) {
        uint64_t c = count_range(A, B, nthreads);
        u128 c2 = c;
        if (maxc < c) c2 = maxc;
        char tmp[48];
        tmp[sizeof tmp - 1] = 0;
        puts(u128_dec(tmp + sizeof tmp - 1, c2));
        return finish_output(0);
    }
    if (!haveB && !qty && isatty(STDOUT_FILENO)) {
        maxc = 20;
        fprintf(stderr, "(showing the first 20 candidates; use -q N, an end B, or redirect the output)\n");
    }
    cand_output = 1;
    print_range(A, B, maxc);
    return cand_finish(0);
}
