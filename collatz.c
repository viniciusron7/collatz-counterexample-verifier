/*
 * collatz.c — fast verifier of counterexamples to the Collatz conjecture.
 *
 * Conjecture: for every integer n ≥ 1, the sequence
 *     x_0 = n,   x_{k+1} = x_k / 2      (x_k even)
 *                x_{k+1} = 3·x_k + 1    (x_k odd)
 * reaches 1. A counterexample would be an n whose trajectory diverges to infinity
 * or enters a cycle other than 1 → 4 → 2 → 1.
 *
 * For a given n (of any size) the program computes, in steps of the rule above:
 *   σ(n)  stopping time: the first k ≥ 1 with x_k < n. Proves that n is not the SMALLEST
 *         counterexample — the criterion of exhaustive searches (n = 1, 2, 3, ... in order);
 *   2^71  the first k with x_k < 2^71. Every integer below 2^71 has already been verified
 *         (D. Barina, J. Supercomputing 81, 2025), so this already proves that n converges;
 *   δ(n)  total stopping time: the first k with x_k = 1 — a complete, self-contained proof.
 *
 * Why it is fast:
 *   1. It uses the accelerated map T(x) = x/2 (even) or (3x+1)/2 (odd). Each odd step
 *      of T is worth 2 Collatz steps: Collatz_steps = T_steps + number of odd steps.
 *   2. Up to 128 bits it uses native arithmetic. Far from the thresholds, 8 steps per table
 *      lookup: T^8(x) = 3^c·⌊x/2^8⌋ + T^8(x mod 2^8). Near them, exact "runs": if
 *      x+1 = 2^a·m (m odd), the next a steps are odd and lead to x ← 3^a·m − 1;
 *      then b = ctz(x) even steps lead to x ← x / 2^b. Below 2^12, the time to 1
 *      comes from a table.
 *   3. For large numbers (GMP) it uses the 2-adic identity
 *          T^k(q·2^k + r) = 3^c·q + T^k(r),   c = number of odd steps of r,
 *      applied recursively (divide and conquer): k steps cost
 *      O(M(k)·log k) instead of O(k²), where M is the cost of GMP multiplication. The power
 *      3^c of each piece comes from the smaller pieces (one square and a small correction).
 *      Pieces of up to 16384 steps advance in blocks of 64 steps, each block a single
 *      multiplication pass over the limbs.
 *   4. Safe blocks: since T(x) ≥ x/2, if x has L bits and the threshold t has B bits, the
 *      next L−1−B steps cannot fall below t and go in a single block. Near the
 *      threshold, watched blocks: since each odd step multiplies x by more than 3/2,
 *      log2(x_j) ≥ log2(x) + c·log2(3) − j. This bound, tracked step by step in fixed
 *      point and always rounded down, tells how far it is safe to advance without skipping
 *      the first instant at which x_k < t. Where it guarantees nothing, the steps are exact.
 *   5. Very large products are split across the performance cores (threads).
 *   6. Nontrivial cycles are detected (Brent's algorithm). If one appears, n IS a
 *      counterexample. Divergence to infinity cannot be proved by computation.
 *
 * Record: every n in [2^71, 2^127) with verdict NO is stored in Tested/ (next to the
 * executable, or in $COLLATZ_TESTED; format in tested.h), and the candidates generator does
 * not list it again. With --data, the trajectory data of each one is stored too (data.csv).
 */

#include <stdio.h>   /* before gmp.h, so that mpz_out_str is declared */
#include <gmp.h>
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <poll.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/stat.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#include "tested.h"            /* format of the Tested/ directory (shared with candidates) */

#if GMP_LIMB_BITS != 64 || GMP_NAIL_BITS != 0
#error "This program requires 64-bit limbs without nails (GMP_LIMB_BITS == 64)."
#endif
_Static_assert(sizeof(unsigned long) == 8, "Requires a 64-bit unsigned long (LP64).");
#ifdef __FAST_MATH__
#error "Do not compile with -ffast-math/-Ofast: the log2 bound of watched blocks depends on an exact log2()."
#endif

__extension__ typedef unsigned __int128 u128;
__extension__ typedef __int128 s128;

#define VERIFIED_BITS  71                  /* every integer in [1, 2^71) converges (Barina, 2025) */
#define BASE_K         64                  /* T steps per block in the base case */
#define MAX_INPUT_BITS ((size_t)1 << 33)   /* sanity limit on the input (~2.6 billion digits) */
#define MAX_DEPTH      2000                /* nested parentheses and powers in the input */

/* Performance parameters (measured on an Apple M5; they change only the time, never results).
   SIMPLE_K and PAR_LIMBS can be overridden with -D (the tests use -DPAR_LIMBS=64 to
   exercise the parallel products with small numbers). */
#ifndef SIMPLE_K
#define SIMPLE_K   16384   /* up to here, k steps in blocks of 64 (cost O(k·L)) beat dividing */
#endif
#define SIMPLE_K1  256     /* with k this small, blocks of 64 even if x is huge */
#define SPLIT_NOPOW 1.5    /* when 3^c is not needed, split x = q·2^k + r only if q has ≥ 1.5·k bits */
#define WATCH_MIN  64      /* initial size of a watched block (doubles while it succeeds) */
#ifndef PAR_LIMBS
#define PAR_LIMBS  1000    /* products with ≥ 1000 limbs (64 000 bits) are done in parallel */
#endif
#define PAR_MAX    4       /* pieces per parallel product (threads, counting the main one) */

/* ------------------------------------------------------------------------- */
/* Options                                                                   */

#define NO_LIMIT UINT64_MAX
static int      opt_stop_only  = 0;          /* -s: stop at the stopping time            */
static int      opt_verbose    = 0;          /* -v                                       */
static int      opt_raw        = 0;          /* -r: compact output                       */
static uint64_t opt_max_steps  = NO_LIMIT;   /* -n: limit on Collatz steps               */
static int      opt_progress   = 0;          /* progress on standard error (terminal)    */
static int      opt_simple     = 0;          /* --simple: no divide and conquer          */
static int      opt_no_native  = 0;          /* --no-native: GMP only                    */
static int      opt_no_blocks  = 0;          /* --no-blocks: exact steps only            */
static int      opt_no_threads = 0;          /* --no-threads: products on the main thread only */
static int      opt_no_record  = 0;          /* --no-record: do not write to Tested/     */

/* Stop requested by a signal (Ctrl+C, kill, closed terminal, closed output). The 1st signal only
   marks the request: the program finishes the output line in progress, discards the number that
   was being computed (not proved, so not recorded), writes and sorts the tested numbers and only
   then exits, by the same signal. A 2nd signal exits immediately (see on_signal). */
static volatile sig_atomic_t stop_sig = 0;
static volatile sig_atomic_t sig_count = 0;      /* stop signals received */
static volatile sig_atomic_t fast_exit_ok = 0;   /* computing a huge number: may exit immediately */
static volatile sig_atomic_t pending_die = 0;    /* signal asking for immediate exit, deferred by a
                                                    short critical section (see on_signal) */
static void die_now(int sig);

/* Compact output (-r) in its own buffer, always with complete lines, written with write().
   When stopping, each write() carries only whole lines (on a pipe, at most PIPE_BUF bytes, which
   makes it atomic); so whoever reads the output never receives half a line. */
static char raw_buf[1 << 16];
static size_t raw_len;
static volatile sig_atomic_t raw_writing = 0;
static int out_is_pipe = 0;

static void raw_flush(void)
{
    size_t off = 0, len = raw_len;
    if (!len) return;
    raw_writing = 1;
    while (off < len) {
        size_t chunk = len - off;
        if ((stop_sig || pending_die) && out_is_pipe && chunk > PIPE_BUF) {
            chunk = PIPE_BUF;                     /* cut at the last '\n' within PIPE_BUF */
            while (chunk > 1 && raw_buf[off + chunk - 1] != '\n') chunk--;
        }
        ssize_t w = write(STDOUT_FILENO, raw_buf + off, chunk);
        if (w < 0) {
            if (errno == EINTR) continue;
            break;                                /* output closed: the rest is lost, whole */
        }
        off += (size_t)w;
        if (pending_die && raw_buf[off - 1] == '\n') break;   /* exit, but at a complete line */
    }
    raw_len = 0;
    raw_writing = 0;
    if (pending_die) die_now(pending_die);
}

static void raw_line(const char *s, size_t n)
{
    if (raw_len + n > sizeof raw_buf) raw_flush();
    memcpy(raw_buf + raw_len, s, n);
    raw_len += n;
}

static void out_sync(void)                       /* deliver what is ready, on both outputs */
{
    raw_flush();
    fflush(stdout);
}

/* ------------------------------------------------------------------------- */
/* Tables and utilities                                                      */

/* log2 bounds in Q64.64 fixed point, always rounded down. */
#define ONE64    ((s128)1 << 64)
#define L3FIX    (ONE64 + (s128)0x95C01A39FBD6879FULL)   /* ⌊log2(3)·2^64⌋ */
#define ODD_GAIN (L3FIX - ONE64)                         /* one odd step: log2(3/2), from below */

#define FIN_BITS 12
static u128     P3[81];            /* 3^i for i ≤ 80 (3^80 < 2^127)  */
static u128     MAXM[81];          /* floor((2^128 − 1) / 3^i)        */
static mpz_t    P3Z[BASE_K + 1];   /* 3^i as mpz, i ≤ 64              */
static uint32_t T8[256];           /* r < 2^8: 3^c | T^8(r) << 13 | c << 26 (3^8 < 2^13) */
static s128     MINPRE8[256];      /* r < 2^8: smallest change of the log2 bound over the 8 steps */
static s128     DELTA8[256];       /* r < 2^8: total change of the bound over the 8 steps */
static uint16_t FIN_T[1 << FIN_BITS], FIN_C[1 << FIN_BITS];   /* x < 2^12: T steps to 1, and odd ones */

static void init_tables(void)
{
    P3[0] = 1;
    for (int i = 1; i <= 80; i++) P3[i] = P3[i - 1] * 3;
    for (int i = 0; i <= 80; i++) MAXM[i] = ~(u128)0 / P3[i];
    for (int i = 0; i <= BASE_K; i++) {
        mpz_init(P3Z[i]);
        mpz_ui_pow_ui(P3Z[i], 3, (unsigned long)i);
    }
    for (uint32_t r = 0; r < 256; r++) {
        uint32_t y = r, c = 0;
        s128 v = 0, mn = 0;
        for (int i = 0; i < 8; i++) {
            if (y & 1) { y = (3 * y + 1) / 2; c++; v += ODD_GAIN; }
            else       { y /= 2; v -= ONE64; }
            if (i == 0 || v < mn) mn = v;
        }
        T8[r] = (uint32_t)P3[c] | (y << 13) | (c << 26);
        MINPRE8[r] = mn;
        DELTA8[r] = v;
    }
    /* Total time of each x < 2^12, in order: follow x until it falls below itself (already computed). */
    for (uint32_t x = 2; x < (1u << FIN_BITS); x++) {
        uint64_t y = x;
        uint32_t k = 0, c = 0;
        while (y >= x) {
            if (y & 1) { y = (3 * y + 1) / 2; c++; }
            else         y /= 2;
            k++;
        }
        FIN_T[x] = (uint16_t)(k + FIN_T[y]);
        FIN_C[x] = (uint16_t)(c + FIN_C[y]);
    }
}

static void bug(const char *what);

static double now_sec(void)
{
    struct timespec ts;
#ifdef CLOCK_MONOTONIC_RAW
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);   /* on macOS, CLOCK_MONOTONIC can go backwards */
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static inline unsigned ctz128(u128 v)   /* v ≠ 0 */
{
    uint64_t lo = (uint64_t)v;
    return lo ? (unsigned)__builtin_ctzll(lo)
              : 64u + (unsigned)__builtin_ctzll((uint64_t)(v >> 64));
}

static inline unsigned bitlen128(u128 v)   /* number of bits (0 for v = 0) */
{
    uint64_t hi = (uint64_t)(v >> 64), lo = (uint64_t)v;
    if (hi) return 128u - (unsigned)__builtin_clzll(hi);
    return lo ? 64u - (unsigned)__builtin_clzll(lo) : 0u;
}

static void mpz_set_u128(mpz_t z, u128 v)
{
    mp_limb_t lo = (mp_limb_t)v, hi = (mp_limb_t)(v >> 64);
    mp_limb_t *p = mpz_limbs_write(z, 2);
    p[0] = lo;
    p[1] = hi;
    mpz_limbs_finish(z, hi ? 2 : (lo ? 1 : 0));
}

static u128 mpz_get_u128(const mpz_t z)   /* assumes 0 ≤ z < 2^128 */
{
    return (u128)mpz_getlimbn(z, 0) | ((u128)mpz_getlimbn(z, 1) << 64);
}

static void mpz_add_u128(mpz_t z, u128 v)   /* z ≥ 0 */
{
    mp_limb_t lo = (mp_limb_t)v, hi = (mp_limb_t)(v >> 64);
    if (!hi) {
        mpz_add_ui(z, z, lo);
        return;
    }
    size_t n = mpz_size(z);
    if (n < 2) {
        u128 s = (u128)mpz_getlimbn(z, 0) + v;
        mp_limb_t carry = s < v;
        mp_limb_t *p = mpz_limbs_write(z, 3);
        p[0] = (mp_limb_t)s;
        p[1] = (mp_limb_t)(s >> 64);
        p[2] = carry;
        mpz_limbs_finish(z, carry ? 3 : (p[1] ? 2 : (p[0] ? 1 : 0)));
        return;
    }
    mp_limb_t *p = mpz_limbs_modify(z, (mp_size_t)n + 1);
    mp_limb_t w[2] = { lo, hi };
    mp_limb_t cy = mpn_add(p, p, (mp_size_t)n, w, 2);
    p[n] = cy;
    mpz_limbs_finish(z, (mp_size_t)n + (cy ? 1 : 0));
}

/* The top 64 bits of z ≥ 1, which has L bits: z = top·2^(L−64) + rest. */
static uint64_t top64(const mpz_t z, size_t L)
{
    if (L <= 64) return (uint64_t)mpz_getlimbn(z, 0) << (64 - L);
    size_t n = mpz_size(z);
    unsigned r = (unsigned)(L % 64);   /* bits used in the top limb (0 = all) */
    mp_limb_t hi = mpz_getlimbn(z, n - 1), lo = mpz_getlimbn(z, n - 2);
    return r ? (hi << (64 - r)) | (lo >> r) : hi;
}

/* Lower bound of log2(x) − log2(t), in Q64.64 (x, t ≥ 1). Uses x ≥ ⌊top/2^11⌋·2^(Lx−53)
   and t < (⌊top/2^11⌋ + 1)·2^(Lt−53); both factors fit exactly in a double. */
static s128 log2_gap_lower(const mpz_t x, const mpz_t t)
{
    size_t Lx = mpz_sizeinbase(x, 2), Lt = mpz_sizeinbase(t, 2);
    double ax = (double)(top64(x, Lx) >> 11), bt = (double)((top64(t, Lt) >> 11) + 1);
    double frac = log2(ax) - log2(bt) - 1e-9;   /* margin ≫ error of log2 (~1e-14) */
    return (s128)((int64_t)Lx - (int64_t)Lt) * ONE64 + (s128)(int64_t)floor(ldexp(frac, 62)) * 4;
}

/* ------------------------------------------------------------------------- */
/* Parallel multiplication                                                   */
/* For inputs of millions of bits, the large products of divide and conquer dominate
   the time. A large product a·b is split into pieces (a = a1·B + a0, b = b1·B + b0, or only
   a in 4 parts if b is much smaller) that helper threads multiply at the same time; GMP
   is reentrant and each piece has its own mpz_t. The result equals the serial product. */

static struct {
    int             ready, nth;    /* initialized? / helper threads created */
    pthread_mutex_t mu;
    pthread_cond_t  go, done;
    mpz_t           va[PAR_MAX], vb[PAR_MAX];   /* read-only views of the pieces */
    mpz_t           r[PAR_MAX];                 /* products of the pieces */
    size_t          off[PAR_MAX];               /* position of each product, in limbs */
    int             njob, next, ndone;
    unsigned        gen;                        /* changes with each batch of pieces */
} PQ = { .mu = PTHREAD_MUTEX_INITIALIZER, .go = PTHREAD_COND_INITIALIZER,
         .done = PTHREAD_COND_INITIALIZER };

/* Take and multiply pieces while there are any (called with the mutex locked). */
static void par_take_jobs(void)
{
    while (PQ.next < PQ.njob) {
        int j = PQ.next++;
        pthread_mutex_unlock(&PQ.mu);
        mpz_mul(PQ.r[j], PQ.va[j], PQ.vb[j]);
        pthread_mutex_lock(&PQ.mu);
        if (++PQ.ndone == PQ.njob) pthread_cond_signal(&PQ.done);
    }
}

static void *par_worker(void *arg)
{
    (void)arg;
    unsigned seen = 0;
    pthread_mutex_lock(&PQ.mu);
    for (;;) {
        while (PQ.gen == seen) pthread_cond_wait(&PQ.go, &PQ.mu);
        seen = PQ.gen;
        par_take_jobs();
    }
    return NULL;
}

static int count_fast_cores(void)
{
    int n = 0;
#if defined(__APPLE__)
    size_t sz = sizeof n;   /* performance cores only: efficiency cores would delay the batch */
    if (sysctlbyname("hw.perflevel0.logicalcpu", &n, &sz, NULL, 0) != 0) n = 0;
#endif
    if (n <= 0) n = (int)sysconf(_SC_NPROCESSORS_ONLN);
    return n;
}

static void par_init(void)
{
    PQ.ready = 1;
    for (int j = 0; j < PAR_MAX; j++) mpz_init(PQ.r[j]);
    int want = opt_no_threads ? 1 : count_fast_cores();
    if (want > PAR_MAX) want = PAR_MAX;
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setstacksize(&at, (size_t)8 << 20);
    /* Stop signals always go to the main thread (the helpers block them). */
    sigset_t block, old;
    sigemptyset(&block);
    sigaddset(&block, SIGINT);
    sigaddset(&block, SIGTERM);
    sigaddset(&block, SIGHUP);
    sigaddset(&block, SIGPIPE);
    pthread_sigmask(SIG_BLOCK, &block, &old);
    for (int i = 1; i < want; i++) {
        pthread_t th;
        if (pthread_create(&th, &at, par_worker, NULL) != 0) break;
        pthread_detach(th);
        PQ.nth++;
    }
    pthread_sigmask(SIG_SETMASK, &old, NULL);
    pthread_attr_destroy(&at);
}

/* r ← a·b, in parallel when the product is large. r must not be a or b. */
static void pmul(mpz_t r, const mpz_t a, const mpz_t b)
{
    size_t na = mpz_size(a), nb = mpz_size(b);
    if (na + nb < PAR_LIMBS || na < 32 || nb < 32) {   /* small, or linear (tiny factor) */
        mpz_mul(r, a, b);
        return;
    }
    if (!PQ.ready) par_init();
    if (PQ.nth == 0) {
        mpz_mul(r, a, b);
        return;
    }
    int sq = a == b;
    if (na < nb) {
        mpz_srcptr t = a; a = b; b = t;
        size_t tn = na; na = nb; nb = tn;
    }
    const mp_limb_t *pa = mpz_limbs_read(a), *pb = mpz_limbs_read(b);
    int nj;
    if (sq) {   /* square: a0², a1² and the cross term a0·a1 (added twice) in 2 halves */
        size_t h = na / 2, h2 = (na - h) / 2;
        mpz_roinit_n(PQ.va[0], pa, (mp_size_t)h);
        mpz_roinit_n(PQ.va[1], pa + h, (mp_size_t)(na - h));
        PQ.vb[0][0] = PQ.va[0][0];   /* same pointer: GMP does the square */
        PQ.vb[1][0] = PQ.va[1][0];
        mpz_roinit_n(PQ.va[2], pa, (mp_size_t)h);
        mpz_roinit_n(PQ.vb[2], pa + h, (mp_size_t)h2);
        mpz_roinit_n(PQ.va[3], pa, (mp_size_t)h);
        mpz_roinit_n(PQ.vb[3], pa + h + h2, (mp_size_t)(na - h - h2));
        PQ.off[0] = 0;
        PQ.off[1] = 2 * h;
        PQ.off[2] = h;
        PQ.off[3] = h + h2;
        nj = 4;
    } else if (2 * nb >= na) {        /* similar sizes: 2×2 pieces */
        size_t ha = na / 2, hb = nb / 2;
        for (int j = 0; j < 4; j++) {
            size_t oa = (j & 1) ? ha : 0, ob = (j & 2) ? hb : 0;
            mpz_roinit_n(PQ.va[j], pa + oa, (mp_size_t)((j & 1) ? na - ha : ha));
            mpz_roinit_n(PQ.vb[j], pb + ob, (mp_size_t)((j & 2) ? nb - hb : hb));
            PQ.off[j] = oa + ob;
        }
        nj = 4;
    } else {                          /* b much smaller: a in 4 parts */
        size_t h = (na + 3) / 4;
        for (int j = 0; j < 4; j++) {
            size_t lo = (size_t)j * h < na ? (size_t)j * h : na, hi = lo + h < na ? lo + h : na;
            mpz_roinit_n(PQ.va[j], pa + lo, (mp_size_t)(hi - lo));
            mpz_roinit_n(PQ.vb[j], pb, (mp_size_t)nb);
            PQ.off[j] = lo;
        }
        nj = 4;
    }
    pthread_mutex_lock(&PQ.mu);
    PQ.njob = nj;
    PQ.next = PQ.ndone = 0;
    PQ.gen++;
    pthread_cond_broadcast(&PQ.go);
    par_take_jobs();   /* the main thread works too */
    while (PQ.ndone < PQ.njob) pthread_cond_wait(&PQ.done, &PQ.mu);
    pthread_mutex_unlock(&PQ.mu);

    size_t n = na + nb;
    mp_limb_t *pr = mpz_limbs_write(r, (mp_size_t)n);
    memset(pr, 0, n * sizeof *pr);
    for (int j = 0; j < nj; j++) {
        size_t m = mpz_size(PQ.r[j]);
        const mp_limb_t *pj = mpz_limbs_read(PQ.r[j]);
        for (int rep = 0; m && rep < (sq && j >= 2 ? 2 : 1); rep++)
            if (mpn_add(pr + PQ.off[j], pr + PQ.off[j], (mp_size_t)(n - PQ.off[j]), pj, (mp_size_t)m))
                bug("parallel product");
    }
    while (n > 0 && pr[n - 1] == 0) n--;
    mpz_limbs_finish(r, (mp_size_t)n);
}

/* z ← 3^e. For large e, repeated squaring with pmul (the largest ones in parallel). */
static void pow3(mpz_t z, uint64_t e)
{
    if (e <= BASE_K) {
        mpz_set(z, P3Z[e]);
    } else if (e < 50000 || opt_no_threads) {
        mpz_ui_pow_ui(z, 3, (unsigned long)e);
    } else {
        mpz_t h;
        mpz_init(h);
        pow3(h, e / 2);
        pmul(z, h, h);
        if (e & 1) mpz_mul_ui(z, z, 3);
        mpz_clear(h);
    }
}

/* z ← z · 3^c */
static void mul_pow3(mpz_t z, uint64_t c)
{
    if (c == 0) return;
    if (c <= 40) {                      /* 3^40 < 2^64 */
        mpz_mul_ui(z, z, (unsigned long)P3[c]);
    } else if (c <= 80) {               /* two in-place passes, no copy of z */
        mpz_mul_ui(z, z, (unsigned long)P3[40]);
        mpz_mul_ui(z, z, (unsigned long)P3[c - 40]);
    } else {
        mpz_t p, t;
        mpz_init(p);
        mpz_init(t);
        pow3(p, c);
        pmul(t, z, p);
        mpz_swap(z, t);
        mpz_clear(p);
        mpz_clear(t);
    }
}

/* ------------------------------------------------------------------------- */
/* Core: advancing k steps of T                                              */

/* a consecutive odd steps (requires a ≤ ctz(x+1)):  x ← 3^a·((x+1)/2^a) − 1 */
static void odd_run(mpz_t x, uint64_t a)
{
    mpz_add_ui(x, x, 1);
    mpz_fdiv_q_2exp(x, x, a);
    mul_pow3(x, a);
    mpz_sub_ui(x, x, 1);
}

/* Reference base case (--simple): x ← T^k(x) for 1 ≤ k ≤ 64, step by step.
   Returns the number of odd steps. */
static unsigned advance_base(mpz_t x, unsigned k)
{
    size_t n = mpz_size(x);
    uint64_t lo = (uint64_t)mpz_getlimbn(x, 0);
    /* With 1 limb the whole x is simulated: T^j(x)+1 ≤ (3/2)^j·(x+1) < 2^102.
       With more limbs only r = x mod 2^k is simulated: T^k(r) ≤ 3^k − 1 < 2^102. */
    u128 y = (n <= 1 || k >= 64) ? lo : (lo & ((UINT64_C(1) << k) - 1));
    unsigned c = 0;
    for (unsigned i = 0; i < k; i++) {
        u128 odd = y & 1;
        c += (unsigned)odd;
        y = (y >> 1) + ((-odd) & (y + 1));   /* even: y/2   odd: (3y+1)/2 = y/2 + y + 1 */
    }
    if (n <= 1) {
        mpz_set_u128(x, y);
        return c;
    }
    mpz_fdiv_q_2exp(x, x, k);   /* T^k(q·2^k + r) = 3^c·q + T^k(r) */
    mul_pow3(x, c);
    mpz_add_u128(x, y);
    return c;
}

/* y ← T^k(y) for k ≤ 64 and y < 2^64 (the result stays < 2^102): 8 steps per table
   lookup, T^8(y) = 3^c·⌊y/2^8⌋ + T^8(y mod 2^8), and the rest step by step. *pc ← number of odd steps. */
static inline u128 sim_steps(u128 y, unsigned k, unsigned *pc)
{
    unsigned c = 0;
    for (; k >= 8; k -= 8) {
        uint32_t e = T8[(uint8_t)y];
        y = (y >> 8) * (e & 0x1fff) + ((e >> 13) & 0x1fff);
        c += e >> 26;
    }
    for (; k > 0; k--) {
        u128 odd = y & 1;
        c += (unsigned)odd;
        y = (y >> 1) + ((-odd) & (y + 1));
    }
    *pc = c;
    return y;
}

/* {p, m} += y (m ≥ 1); needs room for m + 2 limbs. Returns the new size. */
static size_t limbs_add_u128(mp_limb_t *p, size_t m, u128 y)
{
    mp_limb_t w[2] = { (mp_limb_t)y, (mp_limb_t)(y >> 64) }, cy;
    if (m == 1) {
        p[1] = 0;
        m = 2;
    }
    cy = mpn_add(p, p, (mp_size_t)m, w, 2);
    p[m] = cy;
    m += cy;
    while (m > 0 && p[m - 1] == 0) m--;
    return m;
}

/* One block of 1 ≤ k ≤ 64 steps on the limbs of x (n ≥ 2 limbs, normalized), in place:
       x ← 3^c·⌊x/2^k⌋ + T^k(x mod 2^k)
   For k = 64 the shift is a whole limb: a single mpn_mul_1 pass with source
   p + 1 (GMP allows rp ≤ up). Needs room for n + 2 limbs. Returns the new size. */
static size_t block_limbs(mp_limb_t *p, size_t n, unsigned k, unsigned *pc)
{
    unsigned c;
    uint64_t lo = p[0];
    u128 y = sim_steps(k < 64 ? lo & ((UINT64_C(1) << k) - 1) : lo, k, &c);
    const mp_limb_t *src = p;
    size_t m = n;
    if (k == 64) {
        src = p + 1;
        m = n - 1;
    } else {
        mpn_rshift(p, p, (mp_size_t)n, k);
        if (p[n - 1] == 0) m--;
    }
    mp_limb_t cy;
    unsigned e = c;
    if (e > 40) {   /* 3^c does not fit in a limb: two passes */
        cy = mpn_mul_1(p, src, (mp_size_t)m, (mp_limb_t)P3[40]);
        p[m] = cy;
        m += cy != 0;
        src = p;
        e -= 40;
    }
    cy = mpn_mul_1(p, src, (mp_size_t)m, (mp_limb_t)P3[e]);
    p[m] = cy;
    m += cy != 0;
    *pc = c;
    return limbs_add_u128(p, m, y);
}

/* Watch over a long advance (watched block). An even step divides x by 2 and an odd one
   multiplies it by more than 3/2, so log2(x_j) ≥ log2(x_0) + c·log2(3) − j (c = number of odd
   steps among the first j). W.v holds this bound for log2(x_j) − log2(t) in Q64.64 fixed
   point, with every term rounded down: while W.v > 0, x_j > t, and no pending threshold has
   been crossed. Steps are counted in trajectory order, including inside the recursion (which
   walks over the low bits of x, with the same parities). An even step is taken only if W.v
   stays > 0; odd steps only increase the bound. Without watching (on = 0) there is no
   limit; j and c are always counted. */
static struct {
    int      on, stop;   /* watching? / stopped before the end because it was near the threshold */
    s128     v;          /* lower bound of log2(x_j) − log2(t) */
    uint64_t j, c;       /* T steps taken in the current advance and how many were odd */
} W;

static inline uint64_t watch_room(void)   /* how many consecutive even steps are still safe */
{
    if (!W.on) return UINT64_MAX;
    if (W.v <= 0) return 0;
    s128 q = (W.v - 1) / ONE64;   /* largest z with W.v − z·2^64 > 0 */
    return q >= (s128)UINT64_MAX ? UINT64_MAX : (uint64_t)q;
}

static inline void watch_count(uint64_t k, uint64_t c)   /* account for k steps with c odd ones */
{
    W.j += k;
    W.c += c;
    if (W.on) W.v += (s128)c * L3FIX - (s128)k * ONE64;
}

/* How many of the next kk ≤ 64 steps the watch allows. The parities come from the low bits
   lo = x mod 2^64 (equal to those of x in the first 64 steps); stops before the first even step
   that would leave the bound ≤ 0. */
static unsigned watch_allowed(uint64_t lo, unsigned kk)
{
    if (!W.on) return kk;
    s128 v = W.v;
    u128 y = lo;
    unsigned i = 0;
    while (i + 8 <= kk && v + MINPRE8[(uint8_t)y] > 0) {
        uint32_t e = T8[(uint8_t)y];
        v += DELTA8[(uint8_t)y];
        y = (y >> 8) * (e & 0x1fff) + ((e >> 13) & 0x1fff);
        i += 8;
    }
    for (; i < kk; i++) {
        if (y & 1) {
            v += ODD_GAIN;
            y = (y >> 1) + y + 1;   /* (3y+1)/2 */
        } else {
            if (v - ONE64 <= 0) break;
            v -= ONE64;
            y >>= 1;
        }
    }
    return i;
}

/* Even run of up to z steps (x = 0 counts as an endless run: T(0) = 0), respecting the
   watch. Returns the steps taken. */
static uint64_t even_run_watched(mpz_t x, uint64_t z)
{
    uint64_t room = watch_room();
    if (z > room) {
        z = room;
        W.stop = 1;
    }
    mpz_fdiv_q_2exp(x, x, z);
    watch_count(z, 0);
    return z;
}

/* a odd steps (a ≤ ctz(x+1)); they only increase x, so they are always safe. */
static void odd_run_watched(mpz_t x, uint64_t a)
{
    odd_run(x, a);
    watch_count(a, a);
}

/* x ← T^j(x) with j ≤ k, in blocks of 64 steps, each in a single pass over the limbs:
   O(k·L) operations, the best for small k and x. Runs of ≥ 64 equal steps are taken
   at once. j < k only if the watch says stop. Returns the number of odd steps. */
static uint64_t advance_simple(mpz_t x, uint64_t k)
{
    uint64_t c = 0;
    while (k > 0 && !W.stop) {
        size_t n = mpz_size(x);
        mp_limb_t lo = n ? mpz_getlimbn(x, 0) : 0;
        if (n == 0 || (k >= 64 && lo == 0)) {
            uint64_t z = n ? mpz_scan1(x, 0) : k;
            k -= even_run_watched(x, z < k ? z : k);
            continue;
        }
        if (k >= 64 && lo == ~(mp_limb_t)0) {
            uint64_t a = mpz_scan0(x, 0);
            if (a > k) a = k;
            odd_run_watched(x, a);
            c += a;
            k -= a;
            continue;
        }
        if (k <= 64 && n > 1024) {
            /* A single block on a large x (steps near a threshold): in-place mpz operations,
               without requesting extra room (which would force reallocating and copying all of x). */
            unsigned kk = watch_allowed(lo, (unsigned)k), cc;
            if (kk < k) W.stop = 1;
            if (kk == 0) break;
            u128 y = sim_steps(kk < 64 ? lo & ((UINT64_C(1) << kk) - 1) : lo, kk, &cc);
            mpz_fdiv_q_2exp(x, x, kk);
            mul_pow3(x, cc);
            mpz_add_u128(x, y);
            c += cc;
            k -= kk;
            watch_count(kk, cc);
            continue;
        }
        size_t cap = n + 8;
        mp_limb_t *p = mpz_limbs_modify(x, (mp_size_t)cap);
        while (k > 0) {
            unsigned kk = watch_allowed(p[0], k >= 64 ? 64 : (unsigned)k), cc;
            if (kk < (k >= 64 ? 64 : k)) W.stop = 1;
            if (kk == 0) break;
            if (n == 1) {
                u128 y = sim_steps(p[0], kk, &cc);
                p[0] = (mp_limb_t)y;
                p[1] = (mp_limb_t)(y >> 64);
                n = p[1] ? 2 : 1;
            } else {
                n = block_limbs(p, n, kk, &cc);
            }
            c += cc;
            k -= kk;
            watch_count(kk, cc);
            if (W.stop) break;
            if (n + 2 > cap) {
                mpz_limbs_finish(x, (mp_size_t)n);
                cap = 2 * n + 8;
                p = mpz_limbs_modify(x, (mp_size_t)cap);
            }
            if (k >= 64 && (p[0] == 0 || p[0] == ~(mp_limb_t)0)) break;   /* long run */
        }
        mpz_limbs_finish(x, (mp_size_t)n);
    }
    return c;
}

/* P ← P·3^c (P = 3^something) */
static void pow3_acc(mpz_t P, uint64_t c)
{
    if (c == 0) return;
    if (mpz_cmp_ui(P, 1) == 0) pow3(P, c);
    else                       mul_pow3(P, c);
}

/* Q ← 3^b, given P = 3^a with a ≈ b: P times or divided (exact division) by 3^|a−b|. */
static void pow3_near(mpz_t Q, const mpz_t P, uint64_t a, uint64_t b, mpz_t tmp)
{
    uint64_t d = a < b ? b - a : a - b;
    mpz_srcptr m;
    if (d <= BASE_K) {
        m = P3Z[d];
    } else {
        mpz_ui_pow_ui(tmp, 3, (unsigned long)d);
        m = tmp;
    }
    if (b >= a)      pmul(Q, P, m);
    else if (d <= 40) mpz_divexact_ui(Q, P, (unsigned long)P3[d]);
    else             mpz_divexact(Q, P, m);
}

/* P = 3^a, Q = 3^b  →  P ← 3^(a+b). With a ≈ b, one square and a small correction,
   (3^min)²·3^|a−b|, costs less than the product. Q and tmp are destroyed. */
static void pow_combine(mpz_t P, uint64_t a, mpz_t Q, uint64_t b, mpz_t tmp)
{
    if (b == 0) return;
    if (a == 0) {
        mpz_swap(P, Q);
        return;
    }
    uint64_t lo = a < b ? a : b, dif = a < b ? b - a : a - b;
    if (lo < 64 || dif > lo / 4) {
        pmul(tmp, P, Q);
    } else {
        mpz_ptr m = a <= b ? P : Q;
        pmul(tmp, m, m);
        mul_pow3(tmp, dif);
    }
    mpz_swap(P, tmp);
}

#define MAXD 200                 /* recursion levels (each level at least halves k or x) */
static mpz_t TR[MAXD], TP[MAXD], TM[MAXD];   /* per-level temporaries, reused */

static void init_temps(void)
{
    for (int d = 0; d < MAXD; d++) mpz_inits(TR[d], TP[d], TM[d], NULL);
}

/* x ← T^j(x) with j ≤ k (j < k only if the watch says stop). Returns the number of odd steps;
   the steps taken are added to W.j. If P ≠ NULL, also P ← 3^c, assembled from the
   powers of the pieces. d = recursion level. */
static uint64_t advance_rec(mpz_t x, uint64_t k, mpz_ptr P, int d)
{
    uint64_t c = 0, jstart = W.j;
    if (d >= MAXD) bug("recursion too deep");
    if (P) mpz_set_ui(P, 1);

    while (k > 0 && !W.stop) {
        /* Shortcuts for long runs (structured inputs, such as 2^N ± 1). */
        if (mpz_sgn(x) == 0) {
            k -= even_run_watched(x, k);
            continue;
        }
        if (mpz_even_p(x)) {
            mp_bitcnt_t z = mpz_scan1(x, 0);
            if (z >= 64) {
                k -= even_run_watched(x, z < k ? z : k);
                continue;
            }
        } else {
            mp_bitcnt_t a = mpz_scan0(x, 0);   /* number of trailing 1s = ctz(x+1) */
            if (a >= 64) {
                if (a > k) a = k;
                odd_run_watched(x, a);
                if (P) pow3_acc(P, a);
                c += a;
                k -= a;
                continue;
            }
        }

        size_t L = mpz_sizeinbase(x, 2);
        if (k <= SIMPLE_K && (L < k + 64 || k <= SIMPLE_K1)) {
            uint64_t cc = advance_simple(x, k);
            if (P) pow3_acc(P, cc);
            c += cc;
            break;
        }

        if (L >= k + 64 && (P || L - k >= SPLIT_NOPOW * (double)k)) {
            /* x = q·2^k + r: the steps depend only on r (k bits), and
               T^j(q·2^k + r) = 3^c·q·2^(k−j) + T^j(r) for j ≤ k. */
            mpz_ptr r = TR[d], pr = TP[d], t = TM[d];
            mpz_fdiv_r_2exp(r, x, k);
            mpz_fdiv_q_2exp(x, x, k);
            /* If P = 3^c is already known for a stretch of similar length (the 1st half), 3^cc
               comes from P with a small correction (|cc − c| ~ √k), and the child builds no powers. */
            uint64_t j0 = W.j, done = j0 - jstart;
            int hint = P && c >= 64 && done <= 2 * k && k <= 2 * done;
            uint64_t cc = advance_rec(r, k, hint ? NULL : pr, d + 1);
            if (hint) {
                if ((cc > c ? cc - c : c - cc) <= c / 4) pow3_near(pr, P, c, cc, t);
                else                                     pow3(pr, cc);
            }
            uint64_t jj = W.j - j0;
            pmul(t, x, pr);
            if (jj < k) mpz_mul_2exp(t, t, k - jj);
            mpz_add(t, t, r);
            mpz_swap(x, t);
            if (P) pow_combine(P, c, pr, cc, t);
            c += cc;
            break;
        }

        /* Split the steps in half; the second half continues in the loop. */
        uint64_t k1 = k / 2, j0 = W.j;
        mpz_ptr p1 = P ? TP[d] : NULL;
        uint64_t c1 = advance_rec(x, k1, p1, d + 1);
        if (P) pow_combine(P, c, p1, c1, TM[d]);
        c += c1;
        k -= W.j - j0;
    }
    return c;
}

/* x ← T^k(x), unwatched. Returns the number of odd steps among the k steps. */
static uint64_t advance(mpz_t x, uint64_t k)
{
    uint64_t c = 0;

    if (opt_simple) {   /* reference: blocks of 64 steps, no recursion or shortcuts */
        while (k > 0) {
            unsigned kk = k > BASE_K ? BASE_K : (unsigned)k;
            c += advance_base(x, kk);
            k -= kk;
        }
        return c;
    }
    W.on = W.stop = 0;
    W.j = W.c = 0;
    c = advance_rec(x, k, NULL, 0);
    if (W.j != k) bug("incomplete advance");
    return c;
}

/* x ← T^j(x) with j ≤ k, stopping before the lower bound no longer guarantees x_j > t, where
   gap = lower bound of log2(x) − log2(t) (see W). Returns the number of odd steps; *pj ← steps taken. */
static uint64_t advance_watched(mpz_t x, uint64_t k, s128 gap, uint64_t *pj)
{
    W.on = 1;
    W.stop = 0;
    W.j = W.c = 0;
    W.v = gap;
    uint64_t c = advance_rec(x, k, NULL, 0);
    W.on = 0;
    *pj = W.j;
    return c;
}

/* ------------------------------------------------------------------------- */
/* Verification state and events                                            */

enum { EV_SIGMA = 0, EV_BOUND = 1, EV_ONE = 2, NEV = 3 };

typedef struct {
    int      active, done;
    mpz_t    t;          /* threshold: the event is the first k with x_k < t */
    size_t   tbits;      /* number of bits of t                              */
    u128     t128;       /* t, when it fits in 128 bits                      */
    uint64_t kT, cT;     /* T steps and how many were odd, at the event      */
    size_t   xbits;      /* number of bits of x_k at the event               */
    mpz_t    xval;       /* x_k at the event (only if it has ≤ 200 bits: it is what gets printed) */
} event_t;

typedef struct {
    mpz_t    n, x;
    uint64_t kT, cT;     /* T steps taken so far and how many were odd */
    event_t  ev[NEV];
    int      final_ev;   /* event that ends the verification */
    int      limit_hit, cycle_found;
    uint64_t cycle_C;    /* period of the cycle found, in Collatz steps */
    uint64_t wlen;       /* size of the next watched block */
    double   next_progress;
    int      progress_shown;
    int      interrupted;  /* stopped midway by a stop request: no verdict */
} state_t;

static inline int pending(const state_t *S, int e)
{
    return S->ev[e].active && !S->ev[e].done;
}

static int pending_mask(const state_t *S)
{
    int m = 0;
    for (int e = 0; e < NEV; e++)
        if (pending(S, e)) m |= 1 << e;
    return m;
}

static inline uint64_t steps_left(uint64_t kT, uint64_t cT)   /* remaining Collatz steps */
{
    uint64_t used = kT + cT;
    return used >= opt_max_steps ? 0 : opt_max_steps - used;   /* without -n: NO_LIMIT − used */
}

static void record_event(state_t *S, int e, uint64_t kT, uint64_t cT, const mpz_t v)
{
    event_t *E = &S->ev[e];
    E->done = 1;
    E->kT = kT;
    E->cT = cT;
    E->xbits = mpz_sizeinbase(v, 2);
    if (E->xbits <= 200) mpz_set(E->xval, v);
}

static void bug(const char *what)
{
    fflush(stdout);
    fprintf(stderr, "collatz: internal error (%s). Please report it.\n", what);
    exit(4);
}

/* Compare ⌊x/2^s⌋ with t > 0 without building the quotient (the top limbs usually suffice). */
static int cmp_shifted(const mpz_t x, uint64_t s, const mpz_t t)
{
    size_t L = mpz_sizeinbase(x, 2), Lt = mpz_sizeinbase(t, 2);
    if (L <= s) return -1;                         /* quotient 0 */
    if (L - s != Lt) return L - s < Lt ? -1 : 1;
    const mp_limb_t *xp = mpz_limbs_read(x), *tp = mpz_limbs_read(t);
    size_t xn = mpz_size(x), q = (size_t)(s / 64);
    unsigned r = (unsigned)(s % 64);
    for (size_t i = mpz_size(t); i-- > 0;) {       /* same number of limbs as t */
        mp_limb_t lo = xp[q + i], hi = q + i + 1 < xn ? xp[q + i + 1] : 0;
        mp_limb_t w = r ? (lo >> r) | (hi << (64 - r)) : lo;
        if (w != tp[i]) return w < tp[i] ? -1 : 1;
    }
    return 0;
}

/* Even run of b steps from x (x even, b ≤ ctz(x)): for each pending event,
   find the first s ∈ [1, b] with x/2^s < t. Since x/2^s decreases and has L−s bits, it suffices
   to test s0 = L − bits(t) and s0 + 1. Invariant: x ≥ t for every pending event. */
static void check_even_run_big(state_t *S, uint64_t b, mpz_t tmp)
{
    size_t L = mpz_sizeinbase(S->x, 2);
    for (int e = 0; e < NEV; e++) {
        if (!pending(S, e)) continue;
        event_t *E = &S->ev[e];
        if (L - b > E->tbits) continue;   /* x/2^b still has more bits than t */
        uint64_t s = L > E->tbits ? (uint64_t)(L - E->tbits) : 1;
        if (cmp_shifted(S->x, s, E->t) >= 0) {
            if (++s > b) continue;
            if (cmp_shifted(S->x, s, E->t) >= 0) bug("threshold crossing (GMP)");
        }
        if (L - s <= 200) {   /* the value is stored only if it is small */
            mpz_fdiv_q_2exp(tmp, S->x, s);
            record_event(S, e, S->kT + s, S->cT, tmp);
        } else {
            E->done = 1;
            E->kT = S->kT + s;
            E->cT = S->cT;
            E->xbits = L - s;
        }
    }
}

/* One GMP iteration: a safe block, a watched block or an exact run. */
static void step_big(state_t *S, mpz_t tmp)
{
    uint64_t rem = steps_left(S->kT, S->cT);
    if (rem == 0) { S->limit_hit = 1; return; }

    size_t L = mpz_sizeinbase(S->x, 2), Lt = 0;
    int top = -1;   /* pending event with the largest threshold */
    for (int e = 0; e < NEV; e++) {
        if (!pending(S, e)) continue;
        if (top < 0 || mpz_cmp(S->ev[e].t, S->ev[top].t) > 0) top = e;
        if (S->ev[e].tbits > Lt) Lt = S->ev[e].tbits;
    }

    if (!opt_no_blocks) {
        uint64_t ksafe = L >= Lt + 2 ? (uint64_t)(L - 1 - Lt) : 0;
        /* Numbers with few limbs: the safe block is cheaper than the watched one. */
        if (ksafe > 0 && (opt_simple || ksafe >= L / 2 || (L <= 512 && ksafe >= 16))) {
            /* Safe block: x_j ≥ x/2^j ≥ 2^(L−1−j) ≥ 2^Lt > t for j ≤ L−1−Lt. */
            uint64_t k = ksafe;
            if (k > rem / 2) {   /* each T step is worth at most 2 Collatz steps */
                k = rem / 2;
                if (k == 0) { S->limit_hit = 1; return; }
            }
            uint64_t c = advance(S->x, k);
            S->kT += k;
            S->cT += c;
            return;
        }
        if (!opt_simple && top >= 0) {
            /* Watched block: continues while the lower bound guarantees x_j > t (see W). On average
               the bound drops ~0.21 bit per step; the size doubles while blocks finish whole
               (trajectories that stay near the threshold for a long time) and goes back to the
               minimum when a block stops before the end. */
            s128 gap = log2_gap_lower(S->x, S->ev[top].t);
            if (gap > 0) {
                uint64_t k = 8 * ksafe + WATCH_MIN;
                if (k < S->wlen) k = S->wlen;
                if (k > L) k = L;
                if (k > rem / 2) k = rem / 2;
                if (k > 0) {
                    uint64_t j, c = advance_watched(S->x, k, gap, &j);
                    S->wlen = j == k ? 2 * k : WATCH_MIN;
                    if (j > 0) {
                        S->kT += j;
                        S->cT += c;
                        return;
                    }
                }
            }
        }
    }

    if (mpz_odd_p(S->x)) {   /* odd steps only increase x: no threshold is crossed */
        uint64_t a = mpz_scan0(S->x, 0);
        if (a > rem / 2) {
            a = rem / 2;
            if (a == 0) { S->limit_hit = 1; return; }
        }
        odd_run(S->x, a);
        S->kT += a;
        S->cT += a;
    } else {
        uint64_t b = mpz_scan1(S->x, 0);
        if (b > rem) b = rem;
        check_even_run_big(S, b, tmp);
        mpz_fdiv_q_2exp(S->x, S->x, b);
        S->kT += b;
    }
}

static inline unsigned first_cross128(u128 x, unsigned b, u128 t, unsigned tb)
{
    unsigned L = bitlen128(x);
    if (L - b > tb) return 0;
    unsigned s = L > tb ? L - tb : 1;
    if ((x >> s) < t) return s;
    return s + 1 <= b ? s + 1 : 0;
}

/* Record the events crossed in the even run x → x/2^b (z = x/2^b < tmax) and recompute tmax.
   Returns 1 if the final event was reached. */
static int native_events(state_t *S, u128 x, unsigned b, uint64_t kT, uint64_t cT, u128 *tmax)
{
    u128 z = x >> b, tm = 0;
    for (int e = 0; e < NEV; e++) {
        if (!pending(S, e)) continue;
        event_t *E = &S->ev[e];
        if (z < E->t128) {
            unsigned s = first_cross128(x, b, E->t128, (unsigned)E->tbits);
            if (!s) bug("threshold crossing (native)");
            E->done = 1;
            E->kT = kT + s;
            E->cT = cT;
            E->xbits = bitlen128(x >> s);
            mpz_set_u128(E->xval, x >> s);
        } else if (E->t128 > tm) {
            tm = E->t128;
        }
    }
    *tmax = tm;
    return S->ev[S->final_ev].done;
}

/* Native loop (x < 2^127 on entry; intermediate values < 2^128).
   Far from the thresholds (x/2^8 ≥ largest pending threshold) none of the next 8 steps can
   cross a threshold, since T(x) ≥ x/2; so 8 steps are taken per lookup in the T8 table.
   Near the thresholds, exact runs with detection of the first crossing.
   Falls back to GMP if a step would overflow 128 bits. */
static void run_native(state_t *S)
{
    u128 x = mpz_get_u128(S->x);
    uint64_t kT = S->kT, cT = S->cT;
    const u128 lim8 = (u128)1 << 120;   /* below this, 8 steps do not overflow 128 bits */

    u128 tmax = 0;   /* largest pending threshold: a crossing is possible only if x/2^b < tmax */
    for (int e = 0; e < NEV; e++)
        if (pending(S, e) && S->ev[e].t128 > tmax) tmax = S->ev[e].t128;

    /* Brent here too: a nontrivial cycle would lie entirely above 2^71 (below that everything
       converges) and could lie entirely below 2^127, never going back to GMP. Each pass
       of the loop takes at least one step, so x == bsaved only happens in a real cycle. */
    u128 bsaved = 0;
    uint64_t bpw = 1, blam = 0, bC = 0;

    for (;;) {
        if (x >> VERIFIED_BITS) {
            if (x == bsaved) {
                S->cycle_found = 1;
                S->cycle_C = kT + cT - bC;
                break;
            }
            if (++blam == bpw) {
                bsaved = x;
                bC = kT + cT;
                bpw <<= 1;
                blam = 0;
            }
        }
        if (!opt_no_blocks && x < (1u << FIN_BITS) && pending_mask(S) == 1 << EV_ONE
            && steps_left(kT, cT) >= (uint64_t)FIN_T[(unsigned)x] + FIN_C[(unsigned)x]) {
            /* Only reaching 1 remains and x is small: the total time comes from the table. */
            kT += FIN_T[(unsigned)x];
            cT += FIN_C[(unsigned)x];
            x = 1;
            event_t *E = &S->ev[EV_ONE];
            E->done = 1;
            E->kT = kT;
            E->cT = cT;
            E->xbits = 1;
            mpz_set_ui(E->xval, 1);
            break;
        }
        if (opt_no_blocks) {
            /* exact runs only (reference) */
        } else if (opt_max_steps == NO_LIMIT) {
            while (x < lim8 && (x >> 8) >= tmax) {
                uint32_t e = T8[(uint8_t)x];
                x = (x >> 8) * (e & 0x1fff) + ((e >> 13) & 0x1fff);
                kT += 8;
                cT += e >> 26;
            }
        } else {
            while (x < lim8 && (x >> 8) >= tmax && steps_left(kT, cT) >= 16) {
                uint32_t e = T8[(uint8_t)x];
                x = (x >> 8) * (e & 0x1fff) + ((e >> 13) & 0x1fff);
                kT += 8;
                cT += e >> 26;
            }
        }
        uint64_t rem = steps_left(kT, cT);
        if (x & 1) {
            u128 y = x + 1;   /* an odd x here is < 2^127: no overflow */
            unsigned a = ctz128(y);
            if (rem / 2 < a) {
                a = (unsigned)(rem / 2);
                if (a == 0) { S->limit_hit = 1; break; }
            }
            u128 m = y >> a;
            if (a > 80 || m > MAXM[a]) break;   /* 3^a·m − 1 does not fit: GMP takes over */
            x = m * P3[a] - 1;
            kT += a;
            cT += a;
        } else {
            unsigned b = ctz128(x);
            if (rem < b) {
                b = (unsigned)rem;
                if (b == 0) { S->limit_hit = 1; break; }
            }
            u128 z = x >> b;
            if (z < tmax && native_events(S, x, b, kT, cT, &tmax)) {
                x = z;
                kT += b;
                break;
            }
            x = z;
            kT += b;
        }
    }
    mpz_set_u128(S->x, x);
    S->kT = kT;
    S->cT = cT;
}

static char *put_str(char *p, const char *s)   /* copy s to p; return the end */
{
    size_t n = strlen(s);
    memcpy(p, s, n);
    return p + n;
}

static char *put_u64(char *p, uint64_t v)   /* write v in decimal; return the end */
{
    char tmp[20];
    int n = 0;
    do {
        tmp[n++] = (char)('0' + v % 10);
        v /= 10;
    } while (v);
    while (n) *p++ = tmp[--n];
    return p;
}

static void fmt_u64(char *buf, uint64_t v)   /* 1234567 → "1,234,567" */
{
    char tmp[32];
    int n = snprintf(tmp, sizeof tmp, "%" PRIu64, v), j = 0;
    for (int i = 0; i < n; i++) {
        buf[j++] = tmp[i];
        int rest = n - 1 - i;
        if (rest > 0 && rest % 3 == 0) buf[j++] = ',';
    }
    buf[j] = 0;
}

static const char *pl(uint64_t v, const char *one, const char *many)   /* "1 step", "2 steps" */
{
    return v == 1 ? one : many;
}

static void progress(state_t *S)
{
    double t = now_sec();
    if (t < S->next_progress) return;
    S->next_progress = t + 1.0;
    char a[40], b[40];
    fmt_u64(a, S->kT + S->cT);
    fmt_u64(b, mpz_sizeinbase(S->x, 2));
    fprintf(stderr, "\r  … %s steps; current x has %s bits        ", a, b);
    S->progress_shown = 1;
}

static void run(state_t *S)
{
    mpz_t tmp, saved;
    mpz_init(tmp);
    mpz_init(saved);
    uint64_t pw = 1, lam = 0, saved_C = 0;
    int saved_mask = -1, warm = 0;
    S->next_progress = now_sec() + 1.0;

    while (!S->ev[S->final_ev].done && !S->limit_hit && !S->cycle_found) {
        if (stop_sig) {                        /* stop requested: abandon this number */
            S->interrupted = 1;
            break;
        }
        if (!opt_no_native && mpz_sizeinbase(S->x, 2) <= 127) {
            int fits = 1;   /* invariant: pending thresholds ≤ x; they must fit in 128 bits */
            for (int e = 0; e < NEV; e++)
                if (pending(S, e) && S->ev[e].tbits > 128) fits = 0;
            if (fits) {
                run_native(S);
                if (S->ev[S->final_ev].done || S->limit_hit || S->cycle_found) break;
                /* Stopped before an odd run that would overflow 128 bits: GMP takes it
                   below, and the whole (native stretch + GMP step) counts as one iteration. */
            }
        }
        step_big(S, tmp);
        if (S->limit_hit) break;   /* no step was taken: x did not change (not a cycle) */

        /* Brent: the sequence of states between iterations is deterministic in x. A nontrivial
           cycle would have billions of steps, so it starts only after 64 iterations with the
           same set of pending events (avoids copying x needlessly on short trajectories). */
        int mask = pending_mask(S);
        if (mask != saved_mask) {
            saved_mask = mask;
            warm = 0;
        } else if (warm < 64) {
            if (++warm == 64) {
                mpz_set(saved, S->x);
                saved_C = S->kT + S->cT;
                pw = 1;
                lam = 0;
            }
        } else if (mpz_cmp(S->x, saved) == 0) {
            if (S->kT + S->cT == saved_C) bug("cycle of period 0");
            S->cycle_found = 1;
            S->cycle_C = S->kT + S->cT - saved_C;
            break;
        } else if (++lam == pw) {
            mpz_set(saved, S->x);
            saved_C = S->kT + S->cT;
            pw <<= 1;
            lam = 0;
        }
        if (opt_progress) progress(S);
    }
    if (S->progress_shown) fprintf(stderr, "\r%78s\r", "");
    mpz_clear(tmp);
    mpz_clear(saved);
}

/* ------------------------------------------------------------------------- */
/* Output                                                                    */

static void fmt_time(char *buf, size_t sz, double s)
{
    if (s < 999.95e-6)       snprintf(buf, sz, "%.1f µs", s * 1e6);   /* the limits avoid "1000.0 µs" */
    else if (s < 0.999995)   snprintf(buf, sz, "%.2f ms", s * 1e3);
    else if (s < 59.9995)    snprintf(buf, sz, "%.3f s", s);
    else {
        unsigned long t = (unsigned long)(s + 0.5);
        if (t < 3600) snprintf(buf, sz, "%lu min %02lu s", t / 60, t % 60);
        else          snprintf(buf, sz, "%lu h %02lu min %02lu s", t / 3600, t / 60 % 60, t % 60);
    }
}

static void free_str(char *s)
{
    void (*freefunc)(void *, size_t);
    mp_get_memory_functions(NULL, NULL, &freefunc);
    freefunc(s, strlen(s) + 1);
}

/* Print z in full if it is small; otherwise, abbreviated. */
static void print_mpz_short(FILE *f, const mpz_t z)
{
    size_t bits = mpz_sizeinbase(z, 2);
    char c[40];
    if (bits <= 200) {
        mpz_out_str(f, 10, z);
    } else if (bits <= 200000) {
        char *s = mpz_get_str(NULL, 10, z);
        size_t len = strlen(s);
        fmt_u64(c, len);
        fprintf(f, "%.12s…%s (%s digits)", s, s + len - 12, c);
        free_str(s);
    } else {
        fmt_u64(c, bits);
        fprintf(f, "a number with %s bits", c);
    }
}

/* Write n bytes of s replacing control characters with '?' (labels come from the input). */
static void put_clean(FILE *f, const char *s, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        fputc(c < 0x20 || c == 0x7F ? '?' : c, f);
    }
}

/* Long labels are abbreviated without cutting UTF-8 characters in half. */
static void print_label(FILE *f, const char *label)
{
    size_t len = strlen(label);
    if (len <= 70) {
        put_clean(f, label, len);
        return;
    }
    size_t a = 30, b = len - 30;
    while (a > 0 && ((unsigned char)label[a] & 0xC0) == 0x80) a--;
    while (b < len && ((unsigned char)label[b] & 0xC0) == 0x80) b++;
    put_clean(f, label, a);
    fputs("…", f);
    put_clean(f, label + b, len - b);
}

static void print_size(FILE *f, const mpz_t n)
{
    size_t bits = mpz_sizeinbase(n, 2), digits;
    int exact = bits <= 20000;
    if (exact) {
        char *s = mpz_get_str(NULL, 10, n);
        digits = strlen(s);
        free_str(s);
    } else {
        digits = mpz_sizeinbase(n, 10);   /* exact or 1 too many */
    }
    char b[40], d[40];
    fmt_u64(b, bits);
    fmt_u64(d, digits);
    fprintf(f, "[%s %s · %s%s %s]", b, pl(bits, "bit", "bits"), exact ? "" : "≈", d,
            pl(digits, "digit", "digits"));
}

static int is_plain_decimal(const char *s)   /* digits only (and grouping spaces/'_') */
{
    int digits = 0;
    for (; *s; s++) {
        if (isdigit((unsigned char)*s)) digits = 1;
        else if (!isspace((unsigned char)*s) && *s != '_') return 0;
    }
    return digits;
}

enum { V_NO, V_NOTMIN, V_UNKNOWN, V_CYCLE };

static int verdict(const state_t *S)
{
    const event_t *Es = &S->ev[EV_SIGMA], *Eb = &S->ev[EV_BOUND], *E1 = &S->ev[EV_ONE];
    if (S->cycle_found)              return V_CYCLE;
    if (E1->active && E1->done)      return V_NO;   /* reached 1 */
    if (Eb->active && Eb->done)      return V_NO;   /* n < 2^71 or fell below 2^71 */
    if (Es->done)                    return V_NOTMIN;
    return V_UNKNOWN;
}

static void report(state_t *S, int v, const char *label, int from_file, double t_parse, double t_run)
{
    const event_t *Es = &S->ev[EV_SIGMA], *Eb = &S->ev[EV_BOUND], *E1 = &S->ev[EV_ONE];
    int small_n = mpz_sizeinbase(S->n, 2) <= VERIFIED_BITS;   /* n < 2^71 */
    int is_one = mpz_cmp_ui(S->n, 1) == 0;

    if (opt_raw) {   /* built by hand: printf weighs when there are millions of lines */
        static const char *code[] = { "NO", "NOTMIN", "UNKNOWN", "CYCLE" };
        char line[256], *p = put_str(line, code[v]);
        if (Es->done) {
            p = put_u64(put_str(p, " sigma="), Es->kT + Es->cT);
            p = put_u64(put_str(p, " sigmaT="), Es->kT);
        } else {
            p = put_str(p, " sigma=- sigmaT=-");
        }
        if (Eb->active && Eb->done) p = put_u64(put_str(p, " bound="), Eb->kT + Eb->cT);
        else                        p = put_str(p, " bound=-");
        if (E1->active && E1->done) {
            p = put_u64(put_str(p, " total="), E1->kT + E1->cT);
            p = put_u64(put_str(p, " totalT="), E1->kT);
        } else {
            p = put_str(p, " total=- totalT=-");
        }
        p = put_u64(put_str(p, " bits="), mpz_sizeinbase(S->n, 2));
        uint64_t us = (uint64_t)(t_run * 1e6 + 0.5);   /* time in seconds with 6 decimals */
        p = put_u64(put_str(p, " time="), us / 1000000);
        *p++ = '.';
        for (uint64_t d = 100000; d > 0; d /= 10) *p++ = (char)('0' + us / d % 10);
        *p++ = '\n';
        raw_line(line, (size_t)(p - line));
        return;
    }

    static int shown = 0;
    if (shown++) printf("\n");   /* blank line between results */

    char a[40], b[40], tm[40];
    uint64_t k;
    printf("n = ");
    if (from_file) {
        if (mpz_sizeinbase(S->n, 2) <= 200000) {
            char *s = mpz_get_str(NULL, 10, S->n);
            size_t len = strlen(s);
            if (len <= 40) fputs(s, stdout);
            else printf("%.12s…%s", s, s + len - 12);
            free_str(s);
            printf(" ");
        }
        printf("(file ");
        print_label(stdout, label);
        printf(")");
    } else {
        print_label(stdout, label);
        if (!is_plain_decimal(label) && mpz_sizeinbase(S->n, 2) <= 128) {
            printf(" = ");
            mpz_out_str(stdout, 10, S->n);
        }
    }
    printf("   ");
    print_size(stdout, S->n);
    printf("\n");

    switch (v) {
    case V_CYCLE:
        fmt_u64(a, S->cycle_C);
        fmt_u64(b, S->kT + S->cT);
        printf("✘ COUNTEREXAMPLE! The trajectory entered a nontrivial cycle\n"
               "  (period of %s %s; detected after %s steps).\n  Value in the cycle: ",
               a, pl(S->cycle_C, "step", "steps"), b);
        print_mpz_short(stdout, S->x);
        printf("\n");
        break;
    case V_NO:
        if (is_one)
            printf("✔ NOT a counterexample: 1 is already in the trivial cycle 1 → 4 → 2 → 1.\n");
        else if (E1->active && E1->done)
            printf("✔ NOT a counterexample: the trajectory reaches 1.\n");
        else if (!small_n)
            printf("✔ NOT a counterexample: the trajectory falls below 2^71, and every integer\n"
                   "  below 2^71 has already been verified (Barina, 2025).\n");
        else
            printf("✔ NOT a counterexample: n < 2^71, and every integer below 2^71 has already\n"
                   "  been verified (Barina, 2025).\n");
        break;
    case V_NOTMIN:
        printf("◐ n is NOT the SMALLEST counterexample: the trajectory falls below n.\n"
               "  Hence n can only be a counterexample if some number smaller than n is too.\n"
               "  In a search in increasing order (every m < n already verified) this suffices,\n"
               "  but on its own it does not prove that n converges.\n");
        break;
    default:
        fmt_u64(a, opt_max_steps);
        fmt_u64(b, S->kT + S->cT);
        printf("? UNKNOWN: the limit of %s %s (-n) was reached before a conclusion\n"
               "  (steps taken: %s).\n", a, pl(opt_max_steps, "step", "steps"), b);
        break;
    }

    if (Es->done) {
        k = Es->kT + Es->cT;
        fmt_u64(a, k);
        printf("  • stopping time          σ(n) = %s %s  (x_%" PRIu64 " = ", a, pl(k, "step", "steps"), k);
        if (Es->xbits <= 130) {
            mpz_out_str(stdout, 10, Es->xval);
        } else if (Es->xbits <= 200) {   /* abbreviated, so the line does not go far past 80 columns */
            char *s = mpz_get_str(NULL, 10, Es->xval);
            size_t len = strlen(s);
            printf("%.12s…%s (%zu digits)", s, s + len - 12, len);
            free_str(s);
        } else {
            fmt_u64(b, Es->xbits);
            printf("a number with %s bits", b);
        }
        printf(" < n)\n");
    } else if (is_one) {
        printf("  • stopping time          σ(1) does not apply (no term is less than 1)\n");
    }
    if (Eb->active && Eb->done && !small_n) {
        k = Eb->kT + Eb->cT;
        fmt_u64(a, k);
        printf("  • below 2^71             after %s %s  (every m < 2^71 has already been verified)\n",
               a, pl(k, "step", "steps"));
    }
    if (E1->active && E1->done) {
        k = E1->kT + E1->cT;
        fmt_u64(a, k);
        printf("  • total stopping time    δ(%s) = %s %s  (x_%" PRIu64 " = 1)\n",
               is_one ? "1" : "n", a, pl(k, "step", "steps"), k);
    }
    if (S->limit_hit && v != V_UNKNOWN) {   /* what the -n limit prevented from being computed */
        const char *miss[3];
        int nm = 0;
        if (Es->active && !Es->done) miss[nm++] = "σ(n)";
        if (Eb->active && !Eb->done) miss[nm++] = "the drop below 2^71";
        if (E1->active && !E1->done) miss[nm++] = "δ(n)";
        if (nm) {
            fmt_u64(a, opt_max_steps);
            printf("  (the limit of %s %s (-n) was reached before: ", a, pl(opt_max_steps, "step", "steps"));
            for (int i = 0; i < nm; i++) printf("%s%s", i ? ", " : "", miss[i]);
            printf(")\n");
        }
    }
    if (v == V_NOTMIN && opt_stop_only)
        fputs("  (For a complete proof, run without -s: the trajectory is followed to 1.)\n", stdout);
    else if (v == V_NOTMIN)
        fputs("  (For a complete proof, increase the limit -n or run without -n.)\n", stdout);
    else if (v == V_UNKNOWN)
        fputs("  (Increase the limit -n or run without -n to continue.)\n", stdout);
    if (opt_verbose) {
        if (Es->done) {
            fmt_u64(a, Es->kT);
            fmt_u64(b, Es->cT);
            printf("    σ in steps of T(x): %s (%s odd)\n", a, b);
        }
        if (E1->active && E1->done) {
            fmt_u64(a, E1->kT);
            fmt_u64(b, E1->cT);
            printf("    δ in steps of T(x): %s (%s odd)\n", a, b);
        }
        fmt_time(tm, sizeof tm, t_parse);
        printf("    input parsing: %s\n", tm);
    }
    fmt_time(tm, sizeof tm, t_run);
    printf("  computation time: %s\n", tm);
}

/* ------------------------------------------------------------------------- */
/* Record of tested numbers                                                  */
/* Every n in [2^71, 2^127) with verdict NO (proved: not a counterexample) goes to the
   Tested/ directory (next to the executable, or $COLLATZ_TESTED), and candidates does not
   generate it again. Below 2^71 everything has already been verified, and candidates does not
   go beyond 2^127. The format is in tested.h: during the run the numbers go to this process's
   journal (tested.journal.<pid>: records with CRC, append-only, without a lock, so writing never
   waits for another process); at the end, everything is merged into tested.bin, which is kept
   in increasing order.

   A number is recorded only after the proof. The buffer is written every ~1 s, at the end and
   also if the process is interrupted (see on_signal).

   With --data, each recorded number also gets a line in data.csv (see traj_calc). */

#define REC_FLUSH_BYTES (64 * 1024)
#define REC_FLUSH_SECS  1.0

static int    opt_data = 0;                  /* --data: also write data.csv */
static unsigned char *rec_buf;               /* ready 20-byte records */
static char  *data_buf;                      /* ready data.csv lines */
static size_t rec_cap, data_cap;
static volatile size_t rec_len, data_len;    /* complete records/lines only */
static volatile sig_atomic_t rec_modifying;  /* rec_note touching the buffers (realloc, copy) */
static int    rec_state;                     /* 0 = nothing yet, 1 = ok, −1 = failed */
static int    rec_errno;                     /* reason for the failure */
static int    rec_used, data_used;           /* something was recorded in this run */
static uint64_t rec_saved;                   /* numbers successfully written in this run */
static int    rec_finished;                  /* rec_finish has already run */
static int    rec_compact_failed;            /* could not sort at the end */
static double rec_last;

/* Write the buffers to this process's journal. System calls only (usable by the signal
   handler); a repeated write is harmless (the duplicates vanish when sorting). */
static void rec_dump(void)
{
    if (rec_state != 1) return;
    size_t n = rec_len, m = data_len;
    atomic_signal_fence(memory_order_seq_cst);
    if (n) tx_tested_append(rec_buf, n, NULL);
    if (m) tx_data_append(data_buf, m, NULL);
}

static void rec_flush(void)
{
    if (rec_state != 1) return;
    size_t n = rec_len, m = data_len;
    atomic_signal_fence(memory_order_seq_cst);
    int e = 0;
    if (n && !tx_tested_append(rec_buf, n, &e)) {
        rec_state = -1;
        rec_errno = e;
        out_sync();
        fprintf(stderr, "collatz: warning: failed to write to %s (%s); numbers verified from"
                        " now on will not be recorded.\n", TX.my_journal, strerror(e));
        return;
    }
    rec_saved += n / TX_REC;
    if (m && !tx_data_append(data_buf, m, &e)) {
        out_sync();
        fprintf(stderr, "collatz: warning: failed to write to %s (%s); data lines from"
                        " now on will not be recorded.\n", TX.my_data, strerror(e));
        opt_data = 0;
    }
    rec_len = data_len = 0;
    rec_last = now_sec();
}

static int rec_wait_cb(double waited)            /* waiting for another process to sort */
{
    static int told;
    if (waited >= 0.5 && !told) {
        told = 1;
        out_sync();
        fprintf(stderr, "collatz: waiting for another process to finish sorting Tested/.\n"
                        "  The proved numbers are already saved; Ctrl+C exits without losing anything.\n");
    }
    return 1;
}

/* End: write the buffer and sort: merge this process's journal (and those of runs that
   died) into tested.bin and data.csv. */
static void rec_finish(void)
{
    if (rec_finished) return;
    rec_finished = 1;
    rec_flush();
    size_t nj = 0, nd = 0;
    tx_jf_t *js = tx_journals(0, &nj), *jd = tx_journals(1, &nd);
    int work = 0, dwork = 0;
    for (size_t i = 0; i < nj; i++) work |= js[i].state != 3;
    for (size_t i = 0; i < nd; i++) dwork |= jd[i].state != 3;
    free(js);
    free(jd);
    struct stat st;
    if (stat(TX.txt, &st) == 0) work = 1;
    char err[PATH_MAX + 160];
    if (work && !tx_compact(0, err, sizeof err, rec_wait_cb)) {
        rec_compact_failed = 1;
        out_sync();
        fprintf(stderr, "collatz: warning: could not sort Tested/tested.bin: %s.\n"
                        "  The numbers remain stored in the journals and still count for candidates.\n", err);
    }
    if (dwork && !tx_data_compact(err, sizeof err, rec_wait_cb)) {
        out_sync();
        fprintf(stderr, "collatz: warning: could not sort Tested/data.csv: %s.\n"
                        "  The lines remain in the data journals.\n", err);
    }
}

/* Immediate exit: write what is already proved and die by the signal. System calls only. */
static void die_now(int sig)
{
    static const char m1[] = "\ncollatz: interrupted. The numbers already verified are saved in"
                             " Tested/ (the next run sorts them).\n";
    static const char m2[] = "\ncollatz: interrupted in the middle of a huge number (discarded: not"
                             " proved).\n  The numbers already verified are saved in Tested/.\n";
    rec_dump();
    if (sig != SIGPIPE) {
        if (fast_exit_ok) { if (write(STDERR_FILENO, m2, sizeof m2 - 1) < 0) { /* nothing to do */ } }
        else if (write(STDERR_FILENO, m1, sizeof m1 - 1) < 0) { /* nothing to do */ }
    }
    signal(sig, SIG_DFL);
    raise(sig);
}

/* Stop signals (Ctrl+C, kill, closed terminal):
     1st  only sets stop_sig; the program stops at a safe point (see main). In the middle of a
          huge number (fast_exit_ok), it exits immediately: the output was already delivered
          before the computation, and the number in progress was not proved.
     2nd  exits immediately, writing what is already proved. If it lands in a short critical
          section (rec_note touching the buffer, or an output write in progress), the exit waits
          for it to finish.
     3rd  exits no matter what (for a stuck output, such as a pipe nobody reads).
   Closed output (SIGPIPE, e.g. "| head"): stops at a safe point, and writes simply fail. */
static void on_signal(int sig)
{
    if (sig == SIGPIPE) {
        signal(SIGPIPE, SIG_IGN);
        if (!stop_sig) stop_sig = SIGPIPE;
        return;
    }
    int c = ++sig_count;
    if (c == 1) {
        stop_sig = sig;
        if (fast_exit_ok && !rec_modifying && !raw_writing) die_now(sig);
        return;
    }
    if (c == 2 && (rec_modifying || raw_writing)) {
        pending_die = sig;
        return;
    }
    if (!rec_modifying) rec_dump();
    signal(sig, SIG_DFL);
    raise(sig);
}

static void signals_init(void)
{
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_signal;
    sa.sa_flags = SA_RESTART;                 /* reads and writes continue after the signal */
    sigemptyset(&sa.sa_mask);
    int sigs[] = { SIGINT, SIGTERM, SIGHUP, SIGPIPE };
    for (size_t i = 0; i < sizeof sigs / sizeof *sigs; i++) {
        struct sigaction old;
        if (sigaction(sigs[i], NULL, &old) == 0 && old.sa_handler == SIG_IGN) continue;  /* respect nohup etc. */
        sigaction(sigs[i], &sa, NULL);
    }
}

static void rec_init(const char *argv0)
{
    tx_init(argv0);
    tx_mkdir_p(TX.dir);                       /* the signal handler must not need to create the directory */
    rec_last = now_sec();
    atexit(rec_finish);
}

static int buf_reserve(void **buf, size_t *cap, size_t need)
{
    if (need <= *cap) return 1;
    size_t c = *cap ? 2 * *cap : 2 * REC_FLUSH_BYTES;
    while (c < need) c *= 2;
    void *nb = realloc(*buf, c);
    if (!nb) return 0;
    *buf = nb;
    *cap = c;
    return 1;
}

/* ------------------------------------------------------------------------- */
/* Data of each tested number (--data), in the Collatz map x → x/2, x → 3x+1 */
/*   k    first k ≥ 1 with x_k < x_0 (the stopping time σ; x_k = x_0 only in a cycle), and how
          many of x_0, ..., x_{k−1} are odd;
     j    first j with x_j = 2^e (from there on only divisions by 2 down to 1); the file holds
          j − k and the odd terms between x_k and x_j. j − k is negative if the trajectory
          reaches a power of 2 before falling below x_0 (then there are no odd terms between);
     m    first index of the maximum x_m of the trajectory, x_m, and the odd terms among
          x_0, ..., x_{m−1}.
   The computation follows the whole trajectory, in runs: if x + 1 = 2^a·u (u odd), the next
   a odd steps lead to 3^a·u − 1, and the largest value of the run is the last 3x + 1 = 2·(3^a·u − 1).
   A power of 2 can only appear as that last 3x + 1. */

typedef struct {
    uint64_t k, ok, j, oj, e, m, om;
    mpz_t xm;
} traj_t;

static int traj_u128(u128 n, traj_t *D)   /* 0 if some value exceeds 2^128 */
{
    u128 x = n, mx = n;
    uint64_t C = 0, O = 0;
    int have_k = 0;
    D->m = D->om = 0;
    if ((n & (n - 1)) == 0) {                 /* n is already a power of 2 */
        D->j = D->oj = 0;
        D->e = ctz128(n);
    } else {
        for (;;) {
            if (x & 1) {
                unsigned a = ctz128(x + 1);   /* x < 2^127, so x + 1 ≤ 2^127 */
                u128 u = (x + 1) >> a;
                if (a > 80 || u > (((u128)1 << 127) / P3[a])) return 0;
                u128 xa = P3[a] * u - 1;      /* x after the a odd steps (even, < 2^127) */
                uint64_t idx = C + 2 * a - 1; /* index of the last 3x + 1 = 2·xa */
                if (2 * xa > mx) { mx = 2 * xa; D->m = idx; D->om = O + a; }
                O += a;
                if ((xa & (xa - 1)) == 0) {   /* 2·xa is the first power of 2 */
                    D->j = idx;
                    D->oj = O;
                    D->e = ctz128(xa) + 1;
                    break;
                }
                C += 2 * a;
                x = xa;
            } else {
                unsigned b = ctz128(x);
                if (!have_k && (x >> b) < n) {
                    unsigned i = 1;
                    while ((x >> i) >= n) i++;
                    D->k = C + i;
                    D->ok = O;
                    have_k = 1;
                }
                C += b;
                x >>= b;
            }
        }
    }
    if (!have_k) {                            /* falls below n in the descent 2^e → 1 */
        uint64_t i = D->j == 0 ? 1 : 0;
        while (i < D->e && ((u128)1 << (D->e - i)) >= n) i++;
        D->k = D->j + i;
        D->ok = D->oj;
    }
    mpz_set_u128(D->xm, mx);
    return 1;
}

static void traj_mpz(const mpz_t n, traj_t *D)   /* the same, with GMP (trajectories above 2^128) */
{
    mpz_t x, t, u;
    mpz_inits(x, t, u, NULL);
    mpz_set(x, n);
    mpz_set(D->xm, n);
    uint64_t C = 0, O = 0;
    int have_k = 0;
    D->m = D->om = 0;
    if (mpz_popcount(n) == 1) {
        D->j = D->oj = 0;
        D->e = mpz_scan1(n, 0);
    } else {
        for (;;) {
            if (mpz_odd_p(x)) {
                mpz_add_ui(t, x, 1);
                uint64_t a = mpz_scan1(t, 0);
                mpz_tdiv_q_2exp(u, t, a);
                mpz_ui_pow_ui(t, 3, a);
                mpz_mul(x, t, u);
                mpz_sub_ui(x, x, 1);          /* xa */
                uint64_t idx = C + 2 * a - 1;
                mpz_mul_2exp(t, x, 1);
                if (mpz_cmp(t, D->xm) > 0) { mpz_set(D->xm, t); D->m = idx; D->om = O + a; }
                O += a;
                if (mpz_popcount(x) == 1) {
                    D->j = idx;
                    D->oj = O;
                    D->e = mpz_scan1(x, 0) + 1;
                    break;
                }
                C += 2 * a;
            } else {
                uint64_t b = mpz_scan1(x, 0);
                mpz_tdiv_q_2exp(t, x, b);
                if (!have_k && mpz_cmp(t, n) < 0) {
                    uint64_t i = 1;
                    for (;; i++) {
                        mpz_tdiv_q_2exp(t, x, i);
                        if (mpz_cmp(t, n) < 0) break;
                    }
                    D->k = C + i;
                    D->ok = O;
                    have_k = 1;
                }
                C += b;
                mpz_tdiv_q_2exp(x, x, b);
            }
        }
    }
    if (!have_k) {
        uint64_t i = D->j == 0 ? 1 : 0;
        for (; i < D->e; i++) {
            mpz_set_ui(t, 1);
            mpz_mul_2exp(t, t, D->e - i);
            if (mpz_cmp(t, n) < 0) break;
        }
        D->k = D->j + i;
        D->ok = D->oj;
    }
    mpz_clears(x, t, u, NULL);
}

static void traj_calc(const mpz_t n, traj_t *D)
{
    if (mpz_sizeinbase(n, 2) > 127 || !traj_u128(mpz_get_u128(n), D)) traj_mpz(n, D);
}

/* Note n (verdict NO). Only the range [2^71, 2^127) goes to Tested/. S carries σ and δ from
   the verifier, to cross-check the data. */
static void rec_note(const mpz_t n, const state_t *S)
{
    if (opt_no_record) return;
    size_t bits = mpz_sizeinbase(n, 2);
    if (bits <= VERIFIED_BITS || bits > 127) return;
    if (rec_state == 0) rec_state = 1;
    if (rec_state != 1) return;
    rec_modifying = 1;
    if (!buf_reserve((void **)&rec_buf, &rec_cap, rec_len + TX_REC)) {
        rec_modifying = 0;
        return;
    }
    tx_encode_rec(rec_buf + rec_len, mpz_get_u128(n));
    atomic_signal_fence(memory_order_seq_cst);   /* the whole record before counting it */
    rec_len = rec_len + TX_REC;
    rec_used = 1;
    rec_modifying = 0;
    if (pending_die) die_now(pending_die);
    if (opt_data) {
        static traj_t D;
        static int D_init = 0;
        if (!D_init) { mpz_init(D.xm); D_init = 1; }
        traj_calc(n, &D);
        /* cross-check with the verifier: σ (and δ, when it went down to 1) */
        const event_t *Es = &S->ev[EV_SIGMA], *E1 = &S->ev[EV_ONE];
        int ok = 1;
        if (Es->done && (Es->kT + Es->cT != D.k || Es->cT != D.ok)) ok = 0;
        if (E1->active && E1->done && (E1->kT + E1->cT != D.j + D.e || E1->cT != D.oj)) ok = 0;
        if (!ok) {
            fflush(stdout);
            fprintf(stderr, "collatz: INTERNAL ERROR: the data of ");
            mpz_out_str(stderr, 10, n);
            fprintf(stderr, " do not match the verifier; the line was not written. Please report it.\n");
        } else {
            size_t need = mpz_sizeinbase(n, 10) + mpz_sizeinbase(D.xm, 10) + 8 * 24;
            rec_modifying = 1;
            if (buf_reserve((void **)&data_buf, &data_cap, data_len + need)) {
                char *p = data_buf + data_len;
                mpz_get_str(p, 10, n);
                p += strlen(p);
                p += sprintf(p, ",%" PRIu64 ",%" PRIu64 ",%" PRId64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",",
                             D.k, D.ok, (int64_t)D.j - (int64_t)D.k, D.oj - D.ok, D.e, D.m);
                mpz_get_str(p, 10, D.xm);
                p += strlen(p);
                p += sprintf(p, ",%" PRIu64 "\n", D.om);
                atomic_signal_fence(memory_order_seq_cst);
                data_len = (size_t)(p - data_buf);
                data_used = 1;
            }
            rec_modifying = 0;
            if (pending_die) die_now(pending_die);
        }
    }
    if (rec_len >= REC_FLUSH_BYTES || data_len >= REC_FLUSH_BYTES || now_sec() - rec_last >= REC_FLUSH_SECS)
        rec_flush();
}

/* ------------------------------------------------------------------------- */
/* Verification of one number                                                */

static int process(const char *label, const mpz_t n, double t_parse, int from_file)
{
    if (mpz_sgn(n) <= 0) {
        out_sync();   /* keeps the order with standard output under 2>&1 */
        fprintf(stderr, "collatz: error in \"");
        print_label(stderr, label);
        fprintf(stderr, "\": ");
        if (!is_plain_decimal(label) && mpz_sizeinbase(n, 2) <= 200) {
            fprintf(stderr, "the value is ");
            mpz_out_str(stderr, 10, n);
            fprintf(stderr, ", which is ");
        }
        fprintf(stderr, "not a positive integer\n");
        if (opt_raw) raw_line("ERROR not_positive\n", 19);
        return 1;
    }

    /* The state is reused across inputs (avoids allocating and freeing for each number). */
    static state_t S;
    static int S_init = 0;
    if (!S_init) {
        mpz_inits(S.n, S.x, NULL);
        for (int e = 0; e < NEV; e++) mpz_inits(S.ev[e].t, S.ev[e].xval, NULL);
        mpz_set_ui(S.ev[EV_BOUND].t, 1);
        mpz_mul_2exp(S.ev[EV_BOUND].t, S.ev[EV_BOUND].t, VERIFIED_BITS);
        mpz_set_ui(S.ev[EV_ONE].t, 2);
        S_init = 1;
    }
    mpz_set(S.n, n);
    mpz_set(S.x, n);
    mpz_set(S.ev[EV_SIGMA].t, n);
    S.kT = S.cT = 0;
    S.limit_hit = S.cycle_found = 0;
    S.cycle_C = 0;
    S.wlen = WATCH_MIN;
    S.next_progress = 0;
    S.progress_shown = 0;
    for (int e = 0; e < NEV; e++) {
        S.ev[e].done = 0;
        S.ev[e].kT = S.ev[e].cT = 0;
    }

    int is_one = mpz_cmp_ui(n, 1) == 0;
    int small_n = mpz_sizeinbase(n, 2) <= VERIFIED_BITS;
    S.ev[EV_SIGMA].active = !is_one;   /* σ(1) does not exist: 1 → 4 → 2 → 1 */
    S.ev[EV_BOUND].active = !opt_stop_only;
    S.ev[EV_ONE].active   = !opt_stop_only || is_one;
    S.final_ev = (opt_stop_only && !is_one) ? EV_SIGMA : EV_ONE;

    for (int e = 0; e < NEV; e++) {
        event_t *E = &S.ev[e];
        E->tbits = mpz_sizeinbase(E->t, 2);
        E->t128 = E->tbits <= 128 ? mpz_get_u128(E->t) : ~(u128)0;
        /* At step 0: x_0 < 2^71 or x_0 < 2 (σ requires k ≥ 1, and x_0 < x_0 is false). */
        if (E->active && e != EV_SIGMA && mpz_cmp(n, E->t) < 0) record_event(&S, e, 0, 0, n);
    }

    double t0 = now_sec();
    S.interrupted = 0;
    if (mpz_sizeinbase(n, 2) > 4096) {           /* may take long: deliver the output first, so */
        out_sync();                              /* that a Ctrl+C midway can exit immediately */
        fast_exit_ok = 1;
    }
    run(&S);
    fast_exit_ok = 0;
    double t_run = now_sec() - t0;
    if (S.interrupted) return 0;                  /* no verdict: nothing is printed or recorded */

    if (opt_stop_only) {
        /* With -s, the drop below 2^71 is noted only if it is already proved: n < 2^71 (step 0)
           or x_σ < 2^71 — in that case at step σ, since before it every x_k ≥ n ≥ 2^71. */
        event_t *Eb = &S.ev[EV_BOUND], *Es = &S.ev[EV_SIGMA];
        Eb->active = 1;
        Eb->done = 0;
        if (small_n) {
            Eb->done = 1;
            Eb->kT = Eb->cT = 0;
        } else if (Es->done && Es->xbits <= VERIFIED_BITS) {
            Eb->done = 1;
            Eb->kT = Es->kT;
            Eb->cT = Es->cT;
        }
    }

    int v = verdict(&S);
    if (v == V_NO) rec_note(n, &S);               /* proved: goes into Tested/ */
    report(&S, v, label, from_file, t_parse, t_run);

    switch (v) {
    case V_CYCLE:   return 3;
    case V_UNKNOWN: return 2;
    case V_NOTMIN:  return opt_stop_only ? 0 : 2;   /* without -s, the requested complete proof is missing */
    default:        return 0;
    }
}

/* ------------------------------------------------------------------------- */
/* Expression parsing: + - * / % ^ ( ), decimal, 0x hexadecimal, 0b binary   */

typedef struct {
    const char *s;      /* normalized text: no spaces or '_', with "**" → "^" */
    const char *orig;   /* original text, for the positions in the messages   */
    size_t      i;
    int         depth;  /* current nesting of parentheses and exponents */
    char        err[240];
} parser_t;

static int p_expr(parser_t *p, mpz_t r);
static int p_unary(parser_t *p, mpz_t r);

static int utf8_len(const unsigned char *u)   /* bytes of the UTF-8 character at u (0 = invalid) */
{
    int len = u[0] < 0x80 ? 1 : (u[0] & 0xE0) == 0xC0 ? 2 : (u[0] & 0xF0) == 0xE0 ? 3
            : (u[0] & 0xF8) == 0xF0 ? 4 : 0;
    for (int k = 1; k < len; k++)
        if ((u[k] & 0xC0) != 0x80) return 0;
    return len;
}

/* Position (in characters, starting at 1) in the original text of index norm_i of the
   normalized text. Repeats exactly the removals of normalize_expr. */
static size_t orig_pos(const char *in, size_t norm_i)
{
    size_t j = 0, col = 0;
    char last = 0;
    for (size_t i = 0; in[i]; i++) {
        unsigned char ch = (unsigned char)in[i];
        if ((ch & 0xC0) != 0x80) col++;
        if (isspace(ch) || ch == '_') continue;
        if (ch == '*' && i > 0 && in[i - 1] == '*' && last == '*') {
            last = '^';
            continue;
        }
        if (j++ == norm_i) return col;
        last = (char)ch;
    }
    return col + 1;   /* end of the input */
}

static int starts(const char *s, const char *lit) { return strncmp(s, lit, strlen(lit)) == 0; }

/* "'x'", "'−' (U+2212)", ... and a hint for common mistakes. */
static void describe_char(const char *s, char *out, size_t sz)
{
    const unsigned char *u = (const unsigned char *)s;
    int len = utf8_len(u);
    char what[64];
    if (len == 0) {
        snprintf(what, sizeof what, "byte 0x%02X (text is not valid UTF-8)", u[0]);
    } else if (len == 1 && (u[0] < 0x20 || u[0] == 0x7F)) {
        snprintf(what, sizeof what, "control character 0x%02X", u[0]);
    } else if (len == 1) {
        snprintf(what, sizeof what, "'%c'", u[0]);
    } else {
        unsigned long cp = u[0] & (0x7Fu >> len);
        for (int k = 1; k < len; k++) cp = (cp << 6) | (u[k] & 0x3Fu);
        snprintf(what, sizeof what, "'%.*s' (U+%04lX)", len, s, cp);
    }
    const char *hint = "";
    if (*s == '.' || *s == ',')
        hint = " — to separate thousands use a space or _ (e.g. 1_000_000)";
    else if (*s == 'e' || *s == 'E')
        hint = " — scientific notation is not accepted: use 10^9 instead of 1e9";
    else if (starts(s, "−") || starts(s, "–"))
        hint = " — use the hyphen - to subtract";
    else if (starts(s, "×") || starts(s, "·") || starts(s, "⋅"))
        hint = " — use * to multiply";
    else if (starts(s, "²") || starts(s, "³"))
        hint = " — use ^ for powers (e.g. 2^2)";
    else if (starts(s, "÷"))
        hint = " — use / to divide";
    else if (starts(s, "\xC2\xA0") || starts(s, "\xE2\x80\xAF") || starts(s, "\xE2\x80\x89"))
        hint = " — special space (copied from a web page?); use a normal space or _";
    else if (starts(s, "’") || starts(s, "'"))
        hint = " — to separate thousands use a space or _";
    snprintf(out, sz, "%s%s", what, hint);
}

static int p_fail_at(parser_t *p, size_t at, const char *msg)
{
    if (!p->err[0]) snprintf(p->err, sizeof p->err, "%s (position %zu)", msg, orig_pos(p->orig, at));
    return 0;
}

static int p_too_big(parser_t *p, size_t at)
{
    p_fail_at(p, at, "result too large");
    size_t n = strlen(p->err);
    snprintf(p->err + n, sizeof p->err - n, " — the limit is 2^33 bits (≈ 2.6 billion digits)");
    return 0;
}

static int p_fail(parser_t *p, const char *msg) { return p_fail_at(p, p->i, msg); }

/* Error at the current character, showing which one it is (or that the input ended). */
static int p_fail_char(parser_t *p, const char *msg)
{
    if (p->err[0]) return 0;
    size_t pos = orig_pos(p->orig, p->i);
    if (!p->s[p->i]) {
        snprintf(p->err, sizeof p->err, "%s (position %zu), but the input ended", msg, pos);
    } else {
        char what[160];
        describe_char(p->s + p->i, what, sizeof what);
        snprintf(p->err, sizeof p->err, "%s (position %zu): %s", msg, pos, what);
    }
    return 0;
}

static int digit_ok(int ch, int base)
{
    if (base == 2)  return ch == '0' || ch == '1';
    if (base == 10) return ch >= '0' && ch <= '9';
    return isxdigit(ch);
}

static int p_number(parser_t *p, mpz_t r)
{
    const char *s = p->s;
    int base = 10;
    if (s[p->i] == '0' && (s[p->i + 1] == 'x' || s[p->i + 1] == 'X')) {
        base = 16;
        p->i += 2;
    } else if (s[p->i] == '0' && (s[p->i + 1] == 'b' || s[p->i + 1] == 'B')) {
        base = 2;
        p->i += 2;
    }
    size_t st = p->i;
    while (s[p->i] && digit_ok((unsigned char)s[p->i], base)) p->i++;
    size_t len = p->i - st;
    if (len == 0)
        return p_fail_char(p, base == 16 ? "hexadecimal digit expected after 0x"
                            : base == 2  ? "binary digit (0 or 1) expected after 0b"
                                         : "number expected");
    double bits_per_digit = base == 2 ? 1.0 : base == 16 ? 4.0 : 3.33;
    if ((double)len * bits_per_digit > (double)MAX_INPUT_BITS) return p_fail_at(p, st, "number too large");
    char *buf = malloc(len + 1);
    if (!buf) return p_fail(p, "out of memory");
    memcpy(buf, s + st, len);
    buf[len] = 0;
    mpz_set_str(r, buf, base);
    free(buf);
    return 1;
}

static int p_atom(parser_t *p, mpz_t r)
{
    if (p->s[p->i] == '(') {
        if (++p->depth > MAX_DEPTH) return p_fail(p, "expression nested too deeply");
        p->i++;
        if (!p_expr(p, r)) return 0;
        if (p->s[p->i] != ')') return p_fail_char(p, "')' expected");
        p->i++;
        p->depth--;
        return 1;
    }
    return p_number(p, r);
}

static int p_power(parser_t *p, mpz_t r)   /* atom ('^' unary)?  — right-associative */
{
    if (!p_atom(p, r)) return 0;
    if (p->s[p->i] != '^') return 1;
    size_t at = p->i++;   /* position of the '^', for the messages */
    if (++p->depth > MAX_DEPTH) return p_fail_at(p, at, "expression nested too deeply");
    mpz_t e;
    mpz_init(e);
    int ok = p_unary(p, e);
    p->depth--;
    if (ok) {
        if (mpz_sgn(e) < 0) {
            ok = p_fail_at(p, at, "negative exponent");
        } else if (mpz_cmpabs_ui(r, 1) <= 0) {   /* base 0, 1 or −1 */
            if (mpz_sgn(e) == 0) mpz_set_ui(r, 1);
            else if (mpz_sgn(r) < 0 && mpz_even_p(e)) mpz_set_ui(r, 1);
        } else if (!mpz_fits_ulong_p(e)) {
            ok = p_fail_at(p, at, "exponent too large");
        } else {
            unsigned long ee = mpz_get_ui(e);
            long ex;
            double m = mpz_get_d_2exp(&ex, r);   /* |r| = |m|·2^ex, 0.5 ≤ |m| < 1 */
            double bits = (double)ee * ((double)ex + log2(fabs(m)));
            if (bits > (double)MAX_INPUT_BITS) {
                ok = p_too_big(p, at);
            } else {
                if (bits > (double)(1 << 20)) {   /* huge power: may take long; deliver the */
                    out_sync();                    /* output first, so a Ctrl+C exits immediately */
                    fast_exit_ok = 1;
                }
                mpz_pow_ui(r, r, ee);
            }
        }
    }
    mpz_clear(e);
    return ok;
}

static int p_unary(parser_t *p, mpz_t r)   /* signs in sequence, without recursion */
{
    int neg = 0;
    while (p->s[p->i] == '-' || p->s[p->i] == '+') {
        if (p->s[p->i] == '-') neg ^= 1;
        p->i++;
    }
    if (!p_power(p, r)) return 0;
    if (neg) mpz_neg(r, r);
    return 1;
}

static int p_term(parser_t *p, mpz_t r)
{
    if (!p_unary(p, r)) return 0;
    for (;;) {
        char op = p->s[p->i];
        if (op != '*' && op != '/' && op != '%') return 1;
        size_t at = p->i++;   /* position of the operator */
        mpz_t t;
        mpz_init(t);
        int ok = p_unary(p, t);
        if (ok) {
            if (op == '*') {
                if ((double)mpz_sizeinbase(r, 2) + (double)mpz_sizeinbase(t, 2) > (double)MAX_INPUT_BITS + 2)
                    ok = p_too_big(p, at);
                else
                    mpz_mul(r, r, t);
            } else if (mpz_sgn(t) == 0) {
                ok = p_fail_at(p, at, "division by zero");
            } else if (op == '/') {
                mpz_fdiv_q(r, r, t);
            } else {
                mpz_fdiv_r(r, r, t);
            }
        }
        mpz_clear(t);
        if (!ok) return 0;
    }
}

static int p_expr(parser_t *p, mpz_t r)
{
    if (!p_term(p, r)) return 0;
    for (;;) {
        char op = p->s[p->i];
        if (op != '+' && op != '-') return 1;
        p->i++;
        mpz_t t;
        mpz_init(t);
        int ok = p_term(p, t);
        if (ok) {
            if (op == '+') mpz_add(r, r, t);
            else mpz_sub(r, r, t);
        }
        mpz_clear(t);
        if (!ok) return 0;
    }
}

/* Remove spaces and '_' and replace "**" (adjacent asterisks) with "^". */
static char *normalize_expr(const char *in)
{
    size_t n = strlen(in), j = 0;
    char *out = malloc(n + 1);
    if (!out) return NULL;
    for (size_t i = 0; i < n; i++) {
        unsigned char ch = (unsigned char)in[i];
        if (isspace(ch) || ch == '_') continue;
        if (ch == '*' && i > 0 && in[i - 1] == '*' && j > 0 && out[j - 1] == '*') {
            out[j - 1] = '^';
            continue;
        }
        out[j++] = (char)ch;
    }
    out[j] = 0;
    return out;
}

/* Outside -f, a space between digits is accepted only as a thousands separator (groups of 3):
   "27 31" is probably two numbers. Returns the (byte) index of the 2nd group, or -1. */
static long digit_space_problem(const char *in)
{
    for (size_t i = 0; in[i]; i++) {
        if (!isdigit((unsigned char)in[i]) || !isspace((unsigned char)in[i + 1])) continue;
        size_t g = 0;   /* size of the digit group ending at i */
        while (g <= i && isdigit((unsigned char)in[i - g])) g++;
        int first = g > i || !isspace((unsigned char)in[i - g]);   /* did not come after a space */
        if (first && g > 3) {
            size_t k = i + 1;
            while (isspace((unsigned char)in[k])) k++;
            if (isdigit((unsigned char)in[k])) return (long)k;
        }
        size_t k = i + 1;
        while (isspace((unsigned char)in[k])) k++;
        if (isdigit((unsigned char)in[k])) {
            size_t d = 0;
            while (isdigit((unsigned char)in[k + d])) d++;
            if (d != 3) return (long)k;
        }
        i = k - 1;
    }
    return -1;
}

static int dotted_thousands(const char *s);

static int parse_input(const char *text, mpz_t out, char *err, size_t errsz, int allow_wrap)
{
    /* "1.000.000": thousands dots in groups of 3, as the program itself prints. */
    if (dotted_thousands(text)) {
        char *t = malloc(strlen(text) + 1);
        if (!t) {
            snprintf(err, errsz, "out of memory");
            return 0;
        }
        size_t j = 0;
        for (const char *c = text; *c; c++)
            if (*c != '.') t[j++] = *c;
        t[j] = 0;
        int ok = parse_input(t, out, err, errsz, allow_wrap);
        free(t);
        return ok;
    }
    /* Fast path for the common case when reading in batch: digits only, up to 38 (< 2^127). */
    size_t len = strspn(text, "0123456789");
    if (len > 0 && len <= 38 && text[len] == 0) {
        u128 v = 0;
        for (size_t i = 0; i < len; i++) v = v * 10 + (unsigned)(text[i] - '0');
        mpz_set_u128(out, v);
        return 1;
    }
    long bad = allow_wrap ? -1 : digit_space_problem(text);
    if (bad >= 0) {
        size_t col = 0;
        for (long i = 0; i <= bad; i++)
            if (((unsigned char)text[i] & 0xC0) != 0x80) col++;
        snprintf(err, errsz, "space between digits (position %zu): two numbers? Use one number "
                 "per line; to separate thousands use groups of 3 (1 000 000) or _", col);
        return 0;
    }
    char *norm = normalize_expr(text);
    if (!norm) {
        snprintf(err, errsz, "out of memory");
        return 0;
    }
    parser_t p;
    memset(&p, 0, sizeof p);
    p.s = norm;
    p.orig = text;
    int ok = 1;
    if (!norm[0]) {
        snprintf(err, errsz, "empty input");
        ok = 0;
    } else if (!p_expr(&p, out)) {
        snprintf(err, errsz, "%s", p.err);
        ok = 0;
    } else if (norm[p.i]) {
        p_fail_char(&p, "unexpected character");
        snprintf(err, errsz, "%s", p.err);
        ok = 0;
    }
    free(norm);
    return ok;
}

static int process_text(const char *text, const char *label, int from_file)
{
    static mpz_t n;
    static int n_init = 0;
    if (!n_init) {
        mpz_init(n);
        n_init = 1;
    }
    char err[320];
    double t0 = opt_verbose ? now_sec() : 0;   /* the parsing time only appears with -v */
    int rc;
    if (!parse_input(text, n, err, sizeof err, from_file)) {
        out_sync();   /* keeps the order with standard output under 2>&1 */
        fprintf(stderr, "collatz: error in \"");
        print_label(stderr, label);
        fprintf(stderr, "\": %s\n", err);
        if (opt_raw) raw_line("ERROR invalid_input\n", 20);
        rc = 1;
    } else {
        rc = process(label, n, opt_verbose ? now_sec() - t0 : 0, from_file);
    }
    fast_exit_ok = 0;
    /* Compact output: flush the buffer at most every 0.2 s (throughput with many small
       numbers, while results of long computations show up promptly, even with | tee). */
    static double last_flush = -1.0;
    double t = now_sec();
    if (!opt_raw || t - last_flush > 0.2) {
        out_sync();
        last_flush = t;
    }
    return rc;
}

static const char *errno_msg(int e)
{
    switch (e) {
    case ENOENT: return "file not found";
    case EACCES: return "no read permission";
    case EISDIR: return "is a directory";
    case ENOMEM: return "out of memory";
    default:     return strerror(e);
    }
}

/* Read the whole file; on error return NULL and the reason in why. */
static char *read_file(const char *path, size_t *lenp, char *why, size_t whysz)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        snprintf(why, whysz, "%s", errno_msg(errno));
        return NULL;
    }
    size_t cap = 1 << 16, len = 0, got;
    char *buf = malloc(cap);
    while (buf && (got = fread(buf + len, 1, cap - len - 1, f)) > 0) {
        len += got;
        if (cap - len - 1 == 0) {
            char *nb = realloc(buf, cap * 2);
            if (!nb) { free(buf); buf = NULL; break; }
            buf = nb;
            cap *= 2;
        }
    }
    int e = ferror(f) ? errno : 0;
    fclose(f);
    if (!buf || e) {
        snprintf(why, whysz, "%s", buf ? errno_msg(e) : "out of memory");
        free(buf);
        return NULL;
    }
    buf[len] = 0;
    *lenp = len;
    return buf;
}

/* Prepare the content of -f: reject null bytes (UTF-16?), replace the BOM and the comments
   (# to the end of the line) with spaces and warn when the lines look like several numbers. */
static int prepare_file(char *buf, size_t len, const char *path, char *why, size_t whysz)
{
    if (memchr(buf, 0, len)) {
        snprintf(why, whysz, "contains null bytes (UTF-16?); save it as UTF-8 text");
        return 0;
    }
    if (len >= 3 && !memcmp(buf, "\xEF\xBB\xBF", 3)) memset(buf, ' ', 3);
    for (char *p = buf; (p = strchr(p, '#')) != NULL; )
        while (*p && *p != '\n') *p++ = ' ';
    size_t lines = 0, first = 0, prev = 0, chars = 0;
    int irregular = 0;
    for (char *p = buf; *p; ) {
        size_t w = 0;
        for (; *p && *p != '\n'; p++)
            if (!isspace((unsigned char)*p)) w++;
        if (*p) p++;
        if (!w) continue;
        if (lines && prev != first) irregular = 1;   /* a middle line of another length */
        if (!lines) first = w;
        prev = w;
        chars += w;
        lines++;
    }
    /* Spaces between digits within a line (outside thousands groups) are suspicious. */
    {
        size_t lineno = 1;
        for (char *p = buf; *p; lineno++) {
            char *e = strchr(p, '\n');
            if (e) *e = 0;
            if (digit_space_problem(p) >= 0) {
                fflush(stdout);
                fprintf(stderr, "collatz: warning: space between digits in \"");
                print_label(stderr, path);
                fprintf(stderr, "\" (line %zu); the spaces were ignored and everything became ONE number.\n", lineno);
                if (e) *e = '\n';
                break;
            }
            if (!e) break;
            *e = '\n';
            p = e + 1;
        }
    }
    /* Number split across lines: all of the same length (the last one may be shorter). */
    if (lines >= 2 && (irregular || prev > first || first < 20)) {
        fflush(stdout);
        fprintf(stderr, "collatz: warning: the %zu lines of \"", lines);
        print_label(stderr, path);
        fprintf(stderr, "\" were joined into ONE number (%zu characters).\n"
                        "  If they were separate numbers, use: collatz < ", chars);
        print_label(stderr, path);
        fprintf(stderr, "\n");
    }
    return 1;
}

static void usage(FILE *f)
{
    fputs(
"Usage: collatz [options] [n ...]\n"
"Check whether n is a counterexample to the Collatz conjecture (n may have\n"
"millions of digits). Steps are those of the rule x → x/2 (even), x → 3x+1 (odd).\n"
"\n"
"  n                   positive integer: decimal, 0x hexadecimal, 0b binary or\n"
"                      an expression with + - * / % ^ (or **) and parentheses;\n"
"                      / and % are integer division and remainder. Thousands\n"
"                      separators: _, dots (1.000.000) and spaces in groups of 3\n"
"                      (1 000 000). E.g.: 2^100000-1   \"(10^500-1)/9\"   1_000_000\n"
"  -s, --stopping      stop at the stopping time σ(n) = first k with x_k < n\n"
"                      (the fastest). Proves that n is not the SMALLEST\n"
"                      counterexample, which suffices in a search in increasing\n"
"                      order; if n < 2^71 or x_σ < 2^71, proves that n is not a\n"
"                      counterexample.\n"
"  -f, --file FILE     read ONE number from FILE; spaces and line breaks are\n"
"                      ignored (for huge numbers split across lines)\n"
"  -n, --max-steps N   give up after N steps (the result may be UNKNOWN)\n"
"  -r, --raw           compact output, one line per number:\n"
"                      VERDICT sigma= sigmaT= bound= total= totalT= bits= time=\n"
"                      VERDICT: NO, NOTMIN (only \"not the smallest\"), UNKNOWN,\n"
"                      CYCLE or ERROR; \"-\" = not reached; bound = first step with\n"
"                      x_k < 2^71 (0 if n < 2^71); sigmaT/totalT: steps of T\n"
"  -v, --verbose       details (steps of the accelerated map T, timings)\n"
"      --data          also write Tested/data.csv: k = σ(n) and the odd terms up\n"
"                      to x_k; j − k and the odd terms up to x_j = 2^e (the first\n"
"                      power of 2); m, x_m (the trajectory maximum) and the odd\n"
"                      terms up to x_m\n"
"      --no-record     do not record verified numbers in Tested/ (see below)\n"
"  -h, --help          show this help\n"
"Without n, read one number per line from standard input (empty lines and\n"
"# comments are ignored). On a terminal, long computations show progress.\n"
"\n"
"Every n in [2^71, 2^127) with verdict NO is stored in Tested/ (in the program's\n"
"directory, or in $COLLATZ_TESTED), and candidates does not generate it again.\n"
"Records are written every ~1 s, at the end and on interruption; at the end,\n"
"tested.bin is left in increasing order (candidates --tested lists its content).\n"
"Ctrl+C (or kill) stops at a safe point: the output ends at the last complete\n"
"line, the number being computed is discarded and the proved ones are saved in\n"
"order; a 2nd Ctrl+C exits immediately (what was proved stays saved).\n"
"\n"
"Without -s, the trajectory is followed to 1 (complete proof). Along the way the\n"
"program shows σ(n), the step at which x_k drops below 2^71 (every smaller\n"
"integer has been verified: Barina, 2025) and the total stopping time\n"
"δ(n) = first k with x_k = 1.\n"
"\n"
"Exit status (with several n, the largest applies):\n"
"  0  the requested conclusion was reached: not a counterexample (with -s, also\n"
"     \"not the smallest counterexample\")\n"
"  1  input error (or no number on standard input)\n"
"  2  the step limit (-n) ran out before the requested conclusion\n"
"  3  counterexample found (nontrivial cycle)\n",
        f);
}

/* "1.000.000": thousands dots in groups of 3 (as the program displays numbers). */
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

/* Value of -n: integer ≥ 0 (1000000, 1_000_000, 1.000.000 or an expression such as 10^6). */
static int parse_steps(const char *s, uint64_t *out)
{
    char buf[64];
    if (dotted_thousands(s) && strlen(s) < sizeof buf) {
        size_t j = 0;
        for (; *s; s++)
            if (*s != '.') buf[j++] = *s;
        buf[j] = 0;
        s = buf;
    }
    mpz_t v;
    mpz_init(v);
    char err[320];
    int ok = parse_input(s, v, err, sizeof err, 0) && mpz_sgn(v) >= 0 && mpz_sizeinbase(v, 2) <= 64;
    if (ok) *out = (uint64_t)mpz_get_ui(v);
    mpz_clear(v);
    return ok;
}

/* Standard input read line by line with poll() + read(), in its own buffer: waiting for more
   input (even in the middle of a line) is always interrupted by a stop request. */
static char  *in_buf;
static size_t in_len, in_pos, in_cap;
static int    in_eof;

static char *next_line(size_t *lenp)            /* NULL: end of input or stop requested */
{
    for (;;) {
        if (stop_sig) return NULL;
        char *nl = in_len > in_pos ? memchr(in_buf + in_pos, '\n', in_len - in_pos) : NULL;
        if (nl || (in_eof && in_pos < in_len)) {
            char *l = in_buf + in_pos;
            size_t n = nl ? (size_t)(nl - l) : in_len - in_pos;
            l[n] = 0;                              /* there is always 1 free byte (see below) */
            in_pos += n + (nl ? 1 : 0);
            *lenp = n;
            return l;
        }
        if (in_eof) return NULL;
        if (in_pos) {
            memmove(in_buf, in_buf + in_pos, in_len - in_pos);
            in_len -= in_pos;
            in_pos = 0;
        }
        if (in_cap - in_len < (1 << 16) + 1) {
            size_t c = in_cap ? 2 * in_cap : (1 << 17);
            char *nb = realloc(in_buf, c);
            if (!nb) { in_eof = 1; continue; }
            in_buf = nb;
            in_cap = c;
        }
        struct pollfd pf = { STDIN_FILENO, POLLIN, 0 };
        int pr = poll(&pf, 1, 0);
        if (pr == 0) {
            if (opt_raw) out_sync();               /* deliver what is computed before waiting */
            while (!stop_sig && (pr = poll(&pf, 1, 200)) == 0) { }
        }
        if (stop_sig) return NULL;
        ssize_t r = read(STDIN_FILENO, in_buf + in_len, in_cap - in_len - 1);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) in_eof = 1;
        else in_len += (size_t)r;
    }
}

int main(int argc, char **argv)
{
    init_tables();
    init_temps();
    opt_progress = isatty(STDERR_FILENO);

    typedef struct { const char *text; int is_file; } input_t;
    input_t *in = calloc((size_t)argc + 1, sizeof *in);
    int nin = 0, only_numbers = 0;
    if (!in) return 1;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        int is_opt = !only_numbers && a[0] == '-' && a[1] && !isdigit((unsigned char)a[1]) && a[1] != '(';
        if (!is_opt) {
            in[nin++].text = a;
            continue;
        }
        if (a[1] != '-' && a[2] && strspn(a + 1, "svrh") == strlen(a + 1)) {   /* -sv, -rs, ... */
            for (const char *c = a + 1; *c; c++) {
                if (*c == 's') opt_stop_only = 1;
                else if (*c == 'v') opt_verbose = 1;
                else if (*c == 'r') opt_raw = 1;
                else { usage(stdout); free(in); return 0; }
            }
            continue;
        }
        if (!strcmp(a, "--")) {
            only_numbers = 1;
        } else if (!strcmp(a, "-s") || !strcmp(a, "--stopping")) {
            opt_stop_only = 1;
        } else if (!strcmp(a, "-v") || !strcmp(a, "--verbose")) {
            opt_verbose = 1;
        } else if (!strcmp(a, "-r") || !strcmp(a, "--raw")) {
            opt_raw = 1;
        } else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(stdout);
            free(in);
            return 0;
        } else if (!strncmp(a, "-n", 2) || !strcmp(a, "--max-steps") || !strncmp(a, "--max-steps=", 12)) {
            const char *name = a[1] == 'n' ? "-n" : "--max-steps", *v;
            if (a[1] == 'n' && a[2])               v = a + 2 + (a[2] == '=');   /* -n5, -n=5 */
            else if (a[1] == '-' && a[11] == '=')  v = a + 12;                  /* --max-steps=5 */
            else                                   v = i + 1 < argc ? argv[++i] : NULL;
            if (!v) {
                fprintf(stderr, "collatz: %s requires a number of steps (e.g. %s 1000000)\n", name, name);
                free(in);
                return 1;
            }
            if (!parse_steps(v, &opt_max_steps)) {
                fprintf(stderr, "collatz: invalid value for %s: \"", name);
                print_label(stderr, v);
                fprintf(stderr, "\" (use an integer ≥ 0, e.g. 1000000, 1_000_000, 1.000.000 or 10^6)\n");
                free(in);
                return 1;
            }
        } else if ((a[1] == 'f' && a[2]) || !strncmp(a, "--file=", 7)) {
            in[nin].text = a[1] == 'f' ? a + 2 : a + 7;
            in[nin++].is_file = 1;
        } else if (!strcmp(a, "-f") || !strcmp(a, "--file")) {
            if (i + 1 >= argc) {
                fprintf(stderr, "collatz: %s requires a file name\n", a);
                free(in);
                return 1;
            }
            in[nin].text = argv[++i];
            in[nin++].is_file = 1;
        } else if (!strcmp(a, "--simple")) {
            opt_simple = 1;
        } else if (!strcmp(a, "--no-native")) {
            opt_no_native = 1;
        } else if (!strcmp(a, "--no-blocks")) {
            opt_no_blocks = 1;
        } else if (!strcmp(a, "--no-threads")) {
            opt_no_threads = 1;
        } else if (!strcmp(a, "--no-record")) {
            opt_no_record = 1;
        } else if (!strcmp(a, "--data")) {
            opt_data = 1;
        } else {
            fprintf(stderr, "collatz: unknown option: ");
            print_label(stderr, a);
            fprintf(stderr, "\n");
            if (!isalpha((unsigned char)a[1]) && !(a[1] == '-' && isalpha((unsigned char)a[2]))) {
                fprintf(stderr, "For an expression that starts with '-', put -- before it: collatz -- '");
                print_label(stderr, a);
                fprintf(stderr, "'\n");
            }
            fprintf(stderr, "Use \"collatz -h\" to see the options.\n");
            free(in);
            return 1;
        }
    }

    if (!opt_no_record) rec_init(argv[0]);
    signals_init();
    struct stat ost;
    out_is_pipe = fstat(STDOUT_FILENO, &ost) == 0 && (S_ISFIFO(ost.st_mode) || S_ISSOCK(ost.st_mode));
    if (!opt_raw) setvbuf(stdout, NULL, _IOLBF, 0);   /* readable output: one whole line at a time */

    int worst = 0;
    if (nin == 0) {
        if (isatty(STDIN_FILENO))
            fprintf(stderr, "Enter one number per line (Ctrl+D to finish):\n");
        char *line;
        size_t ulen;
        ssize_t len;
        long count = 0;
        while ((line = next_line(&ulen))) {
            len = (ssize_t)ulen;
            char *hash = memchr(line, '#', (size_t)len);   /* comment to the end of the line */
            if (hash) {
                *hash = 0;
                len = hash - line;
            }
            while (len > 0 && isspace((unsigned char)line[len - 1])) line[--len] = 0;
            char *s = line;
            if (!strncmp(s, "\xEF\xBB\xBF", 3)) s += 3;   /* BOM (Windows Notepad) */
            while (isspace((unsigned char)*s)) s++;
            if (!*s) continue;
            count++;
            int rc = process_text(s, s, 0);
            if (rc > worst) worst = rc;
        }
        free(in_buf);
        if (!count && !stop_sig) {
            fprintf(stderr, "collatz: no number in the input.\n");
            worst = 1;
        }
    } else {
        for (int i = 0; i < nin && !stop_sig; i++) {
            int rc;
            if (in[i].is_file) {
                char why[200];
                size_t len = 0;
                char *txt = read_file(in[i].text, &len, why, sizeof why);
                if (txt && prepare_file(txt, len, in[i].text, why, sizeof why)) {
                    rc = process_text(txt, in[i].text, 1);
                } else {
                    fflush(stdout);
                    fprintf(stderr, "collatz: could not use the file \"");
                    print_label(stderr, in[i].text);
                    fprintf(stderr, "\": %s\n", why);
                    if (opt_raw) raw_line("ERROR file\n", 11);
                    rc = 1;
                }
                free(txt);
            } else {
                rc = process_text(in[i].text, in[i].text, 0);
            }
            if (rc > worst) worst = rc;
        }
    }
    out_sync();
    free(in);
    /* End (normal or by a stop request): the output ends at the last complete line; write and
       sort the tested numbers. If there was a stop request (even during the final cleanup),
       exit by the same signal: status 130 for Ctrl+C, and a script loop stops too. */
    if (!opt_no_record) rec_finish();
    if (stop_sig) {
        int sig = stop_sig;
        if (sig != SIGPIPE) {
            char a[32];
            fmt_u64(a, rec_saved);
            if (opt_no_record)
                fprintf(stderr, "\ncollatz: interrupted. The output ends at the last complete line.\n");
            else if (rec_state == -1)
                fprintf(stderr, "\ncollatz: interrupted. %s number%s from this run %s saved; the"
                                " following ones could NOT be written to %s (%s).\n",
                        a, rec_saved == 1 ? "" : "s", rec_saved == 1 ? "was" : "were",
                        TX.dir, strerror(rec_errno));
            else
                fprintf(stderr, "\ncollatz: interrupted. %s number%s verified in this run %s"
                                " saved in %s%s; no line was left half-written.\n",
                        a, rec_saved == 1 ? "" : "s", rec_saved == 1 ? "is" : "are", TX.dir,
                        rec_compact_failed ? " (in the journal)" : ", in order");
        }
        signal(sig, SIG_DFL);
        raise(sig);
    }
    return worst;
}
