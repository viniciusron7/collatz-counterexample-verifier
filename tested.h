/*
 * tested.h — storage of the numbers already tested, shared by collatz.c (writer) and
 * candidates.c (reader).
 *
 * Directory Tested/, next to the executable (or the directory $COLLATZ_TESTED):
 *
 *   tested.bin        every tested number, ascending and without repetition. Blocks of up to
 *                     4096 numbers: the first value (16 bytes), then the differences between
 *                     neighbours in LEB128 (~2 bytes per number, against 23 as text). An index
 *                     at the end stores the first and last value of each block, so that only the
 *                     requested range is read. Header, index and blocks carry a CRC-32.
 *   tested.journal.<pid>
 *                     the journal of each running collatz process: 20-byte records
 *                     (number + CRC-32), append-only, without lock (only that process writes
 *                     to it). Writing is fast, never waits for another process and survives
 *                     interruptions. At the end of the run the journal is merged into
 *                     tested.bin, which is rewritten in order (temporary + rename, atomic).
 *   tested.journal    numbers not yet merged from runs already finished (when tested.bin is
 *                     large, it is only rewritten from time to time).
 *   data.csv          (optional, collatz --data) the data of each tested number, ordered by n.
 *   data.journal.<pid>.csv, data.journal.csv
 *                     data lines not yet merged (per process / from finished runs).
 *   .lock             lock (flock) used only to rewrite tested.bin and data.csv.
 *
 * The journal of a process that died (kill -9, power loss) is merged by the next run; until
 * then it is equally valid for the generator.
 *
 * Safety: a number may exclude something from the generator only if it is intact. A truncated
 * record, a wrong CRC or a corrupt block is ignored (the number would be generated again, which
 * is harmless); nothing corrupt is treated as tested.
 */
#ifndef TESTED_H
#define TESTED_H

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

__extension__ typedef unsigned __int128 tx_u128;

#define TX_FN static __attribute__((unused))   /* each program uses only some of the functions */

#define TX_MIN    ((tx_u128)1 << 71)    /* only the range [2^71, 2^127) is stored */
#define TX_LIMIT  ((tx_u128)1 << 127)
#define TX_BLOCK  4096                  /* numbers per block of tested.bin */
#define TX_REC    20                    /* bytes per journal record */
#define TX_HDR    64
#define TX_IDX    56
#define TX_SMALL  ((uint64_t)1 << 22)   /* up to ~4 million: merge the journal on every run */

static const char TX_MAGIC[8] = { 'C', 'L', 'Z', 'T', 'S', 'T', '0', '1' };

static struct {
    char dir[PATH_MAX], bin[PATH_MAX + 32], tmp[PATH_MAX + 32], journal[PATH_MAX + 32],
         lock[PATH_MAX + 32], txt[PATH_MAX + 32], txt_old[PATH_MAX + 48],
         data[PATH_MAX + 32], data_tmp[PATH_MAX + 32], data_journal[PATH_MAX + 32],
         my_journal[PATH_MAX + 48], my_data[PATH_MAX + 48], journal_tmp[PATH_MAX + 48];
    int  lock_fd;
    uint32_t crc[256];
} TX = { .lock_fd = -1 };

/* ------------------------------------------------------------------------- */
/* Utilities                                                                 */

TX_FN uint32_t tx_crc32(const void *data, size_t n, uint32_t crc)   /* CRC-32 (IEEE) */
{
    const unsigned char *p = data;
    crc = ~crc;
    while (n--) crc = TX.crc[(crc ^ *p++) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

TX_FN void tx_put32(unsigned char *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (unsigned char)(v >> (8 * i)); }
TX_FN void tx_put64(unsigned char *p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = (unsigned char)(v >> (8 * i)); }
TX_FN void tx_put128(unsigned char *p, tx_u128 v) { tx_put64(p, (uint64_t)v); tx_put64(p + 8, (uint64_t)(v >> 64)); }
TX_FN uint32_t tx_get32(const unsigned char *p) { uint32_t v = 0; for (int i = 3; i >= 0; i--) v = v << 8 | p[i]; return v; }
TX_FN uint64_t tx_get64(const unsigned char *p) { uint64_t v = 0; for (int i = 7; i >= 0; i--) v = v << 8 | p[i]; return v; }
TX_FN tx_u128 tx_get128(const unsigned char *p) { return (tx_u128)tx_get64(p) | (tx_u128)tx_get64(p + 8) << 64; }

TX_FN int tx_write_all(int fd, const void *buf, size_t len)
{
    const char *p = buf;
    while (len) {
        ssize_t w = write(fd, p, len);
        if (w < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        p += w;
        len -= (size_t)w;
    }
    return 1;
}

TX_FN int tx_read_all(int fd, void *buf, size_t len, off_t off)
{
    char *p = buf;
    while (len) {
        ssize_t r = pread(fd, p, len, off);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return 0;
        p += r;
        off += r;
        len -= (size_t)r;
    }
    return 1;
}

TX_FN int tx_cmp(const void *a, const void *b)
{
    tx_u128 x = *(const tx_u128 *)a, y = *(const tx_u128 *)b;
    return x < y ? -1 : x > y;
}

TX_FN size_t tx_sort_unique(tx_u128 *v, size_t n)
{
    if (n < 2) return n;
    qsort(v, n, sizeof *v, tx_cmp);
    size_t k = 1;
    for (size_t i = 1; i < n; i++)
        if (v[i] != v[k - 1]) v[k++] = v[i];
    return k;
}

TX_FN int tx_mkdir_p(const char *path)
{
    char tmp[PATH_MAX];
    snprintf(tmp, sizeof tmp, "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p != '/') continue;
        *p = 0;
        if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return 0;
        *p = '/';
    }
    return mkdir(tmp, 0755) == 0 || errno == EEXIST;
}

/* Directory: $COLLATZ_TESTED, or Tested/ next to the executable. Initializes the CRC table. */
TX_FN void tx_init(const char *argv0)
{
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        TX.crc[i] = c;
    }
    const char *env = getenv("COLLATZ_TESTED");
    if (env && *env) {
        snprintf(TX.dir, sizeof TX.dir, "%s", env);
    } else {
        char exe[PATH_MAX] = "", real[PATH_MAX], *slash;
#if defined(__APPLE__)
        uint32_t n = sizeof exe;
        if (_NSGetExecutablePath(exe, &n) != 0) exe[0] = 0;
#elif defined(__linux__)
        ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
        exe[n > 0 ? n : 0] = 0;
#endif
        if (!exe[0] && argv0 && strchr(argv0, '/')) snprintf(exe, sizeof exe, "%s", argv0);
        if (exe[0] && realpath(exe, real) && (slash = strrchr(real, '/'))) {
            *slash = 0;
            snprintf(TX.dir, sizeof TX.dir, "%s/Tested", real);
        } else {
            snprintf(TX.dir, sizeof TX.dir, "Tested");
        }
    }
    snprintf(TX.bin, sizeof TX.bin, "%s/tested.bin", TX.dir);
    snprintf(TX.tmp, sizeof TX.tmp, "%s/tested.bin.tmp", TX.dir);
    snprintf(TX.journal, sizeof TX.journal, "%s/tested.journal", TX.dir);
    snprintf(TX.lock, sizeof TX.lock, "%s/.lock", TX.dir);
    snprintf(TX.txt, sizeof TX.txt, "%s/tested.txt", TX.dir);
    snprintf(TX.txt_old, sizeof TX.txt_old, "%s/tested.txt.imported", TX.dir);
    snprintf(TX.data, sizeof TX.data, "%s/data.csv", TX.dir);
    snprintf(TX.data_tmp, sizeof TX.data_tmp, "%s/data.csv.tmp", TX.dir);
    snprintf(TX.data_journal, sizeof TX.data_journal, "%s/data.journal.csv", TX.dir);
    snprintf(TX.my_journal, sizeof TX.my_journal, "%s/tested.journal.%ld", TX.dir, (long)getpid());
    snprintf(TX.journal_tmp, sizeof TX.journal_tmp, "%s/tested.journal.tmp", TX.dir);
    snprintf(TX.my_data, sizeof TX.my_data, "%s/data.journal.%ld.csv", TX.dir, (long)getpid());
}

/* Directory lock. Readers use LOCK_SH; writers, LOCK_EX. create = create the directory. Uses
   only system calls (it is also called from collatz's signal handler). */
TX_FN int tx_lock(int how, int create)
{
    if (TX.lock_fd < 0) {
        if (create && !tx_mkdir_p(TX.dir)) return 0;
        TX.lock_fd = open(TX.lock, O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (TX.lock_fd < 0) return 0;
    }
    while (flock(TX.lock_fd, how) != 0)
        if (errno != EINTR) return 0;
    return 1;
}

TX_FN void tx_unlock(void)
{
    if (TX.lock_fd >= 0) flock(TX.lock_fd, LOCK_UN);
}

/* Exclusive lock without blocking forever: retries every 20 ms. After each failed attempt it
   calls wait_cb(seconds waited); if that returns 0, it gives up. */
TX_FN int tx_lock_wait(int (*wait_cb)(double), char *err, size_t errsz)
{
    if (TX.lock_fd < 0) {
        if (!tx_mkdir_p(TX.dir)) goto fail;
        TX.lock_fd = open(TX.lock, O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (TX.lock_fd < 0) goto fail;
    }
    for (double waited = 0;; waited += 0.02) {
        if (flock(TX.lock_fd, LOCK_EX | LOCK_NB) == 0) return 1;
        if (errno != EWOULDBLOCK && errno != EINTR) goto fail;
        if (wait_cb && !wait_cb(waited)) {
            snprintf(err, errsz, "waiting for the lock was interrupted");
            return 0;
        }
        struct timespec ts = { 0, 20 * 1000 * 1000 };
        nanosleep(&ts, NULL);
    }
fail:
    snprintf(err, errsz, "could not lock %s (%s)", TX.lock, strerror(errno));
    return 0;
}

/* Journals: the shared one (from finished runs) and the per-process ones. */
typedef struct { char path[PATH_MAX + 64]; int state; } tx_jf_t;   /* 0 shared, 1 mine, 2 dead, 3 alive */

/* Lists the journals of numbers (data = 0) or of data lines (data = 1). */
TX_FN tx_jf_t *tx_journals(int data, size_t *np)
{
    *np = 0;
    DIR *dir = opendir(TX.dir);
    if (!dir) return NULL;
    const char *pre = data ? "data.journal." : "tested.journal.", *suf = data ? ".csv" : "";
    const char *shared = data ? "data.journal.csv" : "tested.journal";
    size_t lp = strlen(pre), ls = strlen(suf), n = 0, cap = 0;
    tx_jf_t *v = NULL;
    struct dirent *de;
    while ((de = readdir(dir))) {
        const char *nm = de->d_name;
        size_t ln = strlen(nm);
        int state;
        if (!strcmp(nm, shared)) {
            state = 0;
        } else {
            if (ln <= lp + ls || strncmp(nm, pre, lp) || strcmp(nm + ln - ls, suf)) continue;
            long pid = 0;
            size_t i = lp;
            for (; i < ln - ls; i++) {
                if (nm[i] < '0' || nm[i] > '9' || pid > 100000000) break;
                pid = pid * 10 + (nm[i] - '0');
            }
            if (i != ln - ls || pid <= 0) continue;
            if (pid == (long)getpid()) state = 1;
            else if (kill((pid_t)pid, 0) == 0 || errno == EPERM) state = 3;
            else state = 2;
        }
        if (n == cap) {
            cap = cap ? 2 * cap : 16;
            tx_jf_t *nv = realloc(v, cap * sizeof *nv);
            if (!nv) break;
            v = nv;
        }
        snprintf(v[n].path, sizeof v[n].path, "%s/%s", TX.dir, nm);
        v[n].state = state;
        n++;
    }
    closedir(dir);
    *np = n;
    return v;
}

/* ------------------------------------------------------------------------- */
/* Journal                                                                   */

/* 20-byte record: n (16 bytes, little-endian) + CRC-32 of those 16 bytes. */
TX_FN void tx_encode_rec(unsigned char *p, tx_u128 v)
{
    tx_put128(p, v);
    tx_put32(p + 16, tx_crc32(p, 16, 0x5EED7E57u));
}

/* Appends to this process's journal, without lock (only this process writes to the file).
   First removes a truncated trailing record (binary) or line (text): this only happens if an
   earlier process with the same pid died in the middle of a write. Only system calls (open,
   fstat, pread, ftruncate, write, close): usable from the signal handler. Preserves errno;
   returns 1 if the data was written. */
TX_FN int tx_journal_append(const char *path, const void *buf, size_t len, int binary, int *err_out)
{
    int saved = errno, ok = 0, e = 0;
    if (!len) return 1;
    int fd = open(path, O_RDWR | O_APPEND | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0 && errno == ENOENT && tx_mkdir_p(TX.dir))
        fd = open(path, O_RDWR | O_APPEND | O_CREAT | O_CLOEXEC, 0644);
    if (fd >= 0) {
        struct stat st;
        off_t size = -1;
        if (fstat(fd, &st) == 0) {
            size = st.st_size;
            if (binary && size % TX_REC) {
                size -= size % TX_REC;
                if (ftruncate(fd, size) != 0) size = -1;
            } else if (!binary && size > 0) {
                char c;
                if (pread(fd, &c, 1, size - 1) != 1) size = -1;
                else if (c != '\n') {
                    char tmp[4096];
                    off_t pos = size, cut = -1;
                    while (pos > 0 && cut < 0) {
                        size_t chunk = pos > (off_t)sizeof tmp ? sizeof tmp : (size_t)pos;
                        if (!tx_read_all(fd, tmp, chunk, pos - (off_t)chunk)) break;
                        for (ssize_t i = (ssize_t)chunk - 1; i >= 0; i--)
                            if (tmp[i] == '\n') { cut = pos - (off_t)chunk + i + 1; break; }
                        pos -= (off_t)chunk;
                        if (pos == 0 && cut < 0) cut = 0;
                    }
                    size = cut >= 0 && ftruncate(fd, cut) == 0 ? cut : -1;
                }
            }
        }
        if (size >= 0) {
            ok = tx_write_all(fd, buf, len);
            if (!ok) {
                e = errno;
                if (ftruncate(fd, size) != 0) { /* nothing else to do */ }
            }
        } else {
            e = errno;
        }
        close(fd);
    } else {
        e = errno;
    }
    if (err_out) *err_out = ok ? 0 : (e ? e : EIO);
    errno = saved;
    return ok;
}

TX_FN int tx_tested_append(const unsigned char *recs, size_t len, int *err_out)
{
    return tx_journal_append(TX.my_journal, recs, len, 1, err_out);
}

/* Reads the intact numbers of a journal (complete records with a correct CRC) in [lo, hi). A
   truncated trailing record belongs to a running (or dead) process: it is simply ignored. *bad
   receives the count of rejected records. */
TX_FN tx_u128 *tx_journal_read(const char *path, tx_u128 lo, tx_u128 hi, size_t *np, uint64_t *bad)
{
    *np = 0;
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return NULL;
    struct stat st;
    tx_u128 *v = NULL;
    size_t n = 0, cap = 0;
    if (fstat(fd, &st) == 0 && st.st_size >= TX_REC) {
        size_t nrec = (size_t)st.st_size / TX_REC, chunk = 1 << 16;
        unsigned char *buf = malloc(chunk * TX_REC);
        for (size_t i = 0; buf && i < nrec; i += chunk) {
            size_t c = nrec - i < chunk ? nrec - i : chunk;
            if (!tx_read_all(fd, buf, c * TX_REC, (off_t)(i * TX_REC))) break;
            for (size_t j = 0; j < c; j++) {
                const unsigned char *p = buf + j * TX_REC;
                tx_u128 x = tx_get128(p);
                if (tx_get32(p + 16) != tx_crc32(p, 16, 0x5EED7E57u) || x < TX_MIN || x >= TX_LIMIT) {
                    (*bad)++;
                    continue;
                }
                if (x < lo || x >= hi) continue;
                if (n == cap) {
                    cap = cap ? 2 * cap : 4096;
                    tx_u128 *nv = realloc(v, cap * sizeof *v);
                    if (!nv) { free(buf); free(v); close(fd); return NULL; }
                    v = nv;
                }
                v[n++] = x;
            }
        }
        free(buf);
    }
    close(fd);
    *np = n;
    return v;
}

/* Reads the legacy tested.txt (text format, one number per line), for migration. */
TX_FN tx_u128 *tx_txt_read(tx_u128 lo, tx_u128 hi, size_t *np, uint64_t *bad, int *exists)
{
    *np = 0;
    *exists = 0;
    int fd = open(TX.txt, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return NULL;
    *exists = 1;
    struct stat st;
    char *buf = NULL;
    size_t len = 0;
    if (fstat(fd, &st) == 0 && st.st_size > 0 && (buf = malloc((size_t)st.st_size)))
        len = tx_read_all(fd, buf, (size_t)st.st_size, 0) ? (size_t)st.st_size : 0;
    close(fd);
    tx_u128 *v = NULL;
    size_t n = 0, cap = 0;
    for (size_t i = 0; i < len;) {
        char *nl = memchr(buf + i, '\n', len - i);
        if (!nl) break;                          /* truncated last line: ignored */
        size_t end = (size_t)(nl - buf), e = end;
        if (e > i && buf[e - 1] == '\r') e--;
        const char *l = buf + i;
        size_t m = e - i;
        i = end + 1;
        if (!m || l[0] == '#') continue;
        tx_u128 x = 0;
        int ok = m <= 39;
        for (size_t j = 0; ok && j < m; j++) {
            if (l[j] < '0' || l[j] > '9' || x > (~(tx_u128)0 - 9) / 10) ok = 0;
            else x = x * 10 + (tx_u128)(l[j] - '0');
        }
        if (!ok || x < TX_MIN || x >= TX_LIMIT) { (*bad)++; continue; }
        if (x < lo || x >= hi) continue;
        if (n == cap) {
            cap = cap ? 2 * cap : 4096;
            tx_u128 *nv = realloc(v, cap * sizeof *v);
            if (!nv) { free(v); free(buf); return NULL; }
            v = nv;
        }
        v[n++] = x;
    }
    free(buf);
    *np = n;
    return v;
}

/* ------------------------------------------------------------------------- */
/* tested.bin: reading                                                       */

typedef struct { tx_u128 first, last; uint64_t off; uint32_t size, count, crc; } tx_idx_t;

typedef struct {
    int       fd;                    /* open tested.bin (stays valid if the file is replaced) */
    uint64_t  count, nblocks;
    tx_idx_t *idx;
    int       corrupt;               /* bad header or index: file ignored */
    uint64_t  bad_blocks, bad_recs;
    /* iteration */
    tx_u128   lo, hi;
    size_t    bi;                    /* current block */
    tx_u128  *blk;                   /* decoded values of the current block */
    size_t    bn, bj;
    tx_u128  *pend;                  /* journal (and legacy .txt) in [lo, hi), sorted */
    size_t    npend, pj;
    int       have_last;
    tx_u128   last;
} tx_reader_t;

/* Opens tested.bin (if present) and reads header and index, checking the CRCs. Call with the
   lock held. */
TX_FN void tx_bin_open(tx_reader_t *r)
{
    r->fd = open(TX.bin, O_RDONLY | O_CLOEXEC);
    if (r->fd < 0) return;
    unsigned char h[TX_HDR];
    struct stat st;
    if (fstat(r->fd, &st) != 0 || st.st_size < TX_HDR || !tx_read_all(r->fd, h, TX_HDR, 0)
        || memcmp(h, TX_MAGIC, 8) != 0 || tx_get32(h + 60) != tx_crc32(h, 60, 0)) {
        r->corrupt = 1;
        return;
    }
    r->count = tx_get64(h + 8);
    r->nblocks = tx_get64(h + 16);
    uint64_t ioff = tx_get64(h + 24);
    uint32_t icrc = tx_get32(h + 32);
    if (r->nblocks > (uint64_t)st.st_size / TX_IDX || ioff + r->nblocks * TX_IDX != (uint64_t)st.st_size) {
        r->corrupt = 1;
        return;
    }
    size_t ib = (size_t)(r->nblocks * TX_IDX);
    unsigned char *raw = malloc(ib ? ib : 1);
    r->idx = calloc(r->nblocks ? r->nblocks : 1, sizeof *r->idx);
    if (!raw || !r->idx || (ib && !tx_read_all(r->fd, raw, ib, (off_t)ioff)) || tx_crc32(raw, ib, 0) != icrc) {
        free(raw);
        r->corrupt = 1;
        return;
    }
    for (uint64_t i = 0; i < r->nblocks; i++) {
        const unsigned char *p = raw + i * TX_IDX;
        r->idx[i].first = tx_get128(p);
        r->idx[i].last = tx_get128(p + 16);
        r->idx[i].off = tx_get64(p + 32);
        r->idx[i].size = tx_get32(p + 40);
        r->idx[i].count = tx_get32(p + 44);
        r->idx[i].crc = tx_get32(p + 48);
    }
    free(raw);
}

/* Decodes block i into r->blk (false if the block is corrupt). */
TX_FN int tx_block_load(tx_reader_t *r, size_t i)
{
    const tx_idx_t *e = &r->idx[i];
    r->bn = r->bj = 0;
    if (e->count == 0 || e->count > TX_BLOCK || e->size < 16 || e->size > 16 + 19 * TX_BLOCK) return 0;
    unsigned char *raw = malloc(e->size);
    if (!raw) return 0;
    int ok = tx_read_all(r->fd, raw, e->size, (off_t)e->off) && tx_crc32(raw, e->size, 0) == e->crc;
    if (ok) {
        tx_u128 v = tx_get128(raw);
        size_t p = 16;
        r->blk[r->bn++] = v;
        while (ok && r->bn < e->count) {
            tx_u128 d = 0;
            unsigned sh = 0;
            for (;;) {
                if (p >= e->size || sh > 126) { ok = 0; break; }
                unsigned char c = raw[p++];
                d |= (tx_u128)(c & 0x7F) << sh;
                sh += 7;
                if (!(c & 0x80)) break;
            }
            if (!ok || d == 0) { ok = 0; break; }
            v += d;
            r->blk[r->bn++] = v;
        }
        ok = ok && p == e->size && v == e->last && r->blk[0] == e->first;
    }
    free(raw);
    if (!ok) { r->bn = 0; r->bad_blocks++; }
    return ok;
}

/* Opens a reader of the tested numbers in [lo, hi), ascending and without repetition:
   tested.bin + journal + legacy tested.txt (if not yet migrated). */
TX_FN void tx_reader_open(tx_reader_t *r, tx_u128 lo, tx_u128 hi)
{
    memset(r, 0, sizeof *r);
    r->fd = -1;
    r->lo = lo;
    r->hi = hi;
    r->blk = malloc(TX_BLOCK * sizeof *r->blk);
    struct stat st;
    if (!r->blk || stat(TX.dir, &st) != 0) return;   /* no tested numbers (or out of memory) */
    /* With the lock free, reads a consistent snapshot. If another process is rewriting
       tested.bin, does not wait: at worst a number is returned twice here (harmless). */
    int locked = TX.lock_fd >= 0 || (TX.lock_fd = open(TX.lock, O_RDWR | O_CREAT | O_CLOEXEC, 0644)) >= 0;
    if (locked) locked = flock(TX.lock_fd, LOCK_SH | LOCK_NB) == 0;
    tx_bin_open(r);
    size_t nj = 0, cap = 0, npend = 0;
    tx_jf_t *js = tx_journals(0, &nj);
    for (size_t i = 0; i <= nj; i++) {           /* every journal, then the legacy .txt */
        size_t nd = 0;
        int txt_exists;
        tx_u128 *d = i < nj ? tx_journal_read(js[i].path, lo, hi, &nd, &r->bad_recs)
                            : tx_txt_read(lo, hi, &nd, &r->bad_recs, &txt_exists);
        if (nd) {
            if (npend + nd > cap) {
                cap = (npend + nd) * 2;
                tx_u128 *nv = realloc(r->pend, cap * sizeof *nv);
                if (!nv) { free(d); break; }
                r->pend = nv;
            }
            memcpy(r->pend + npend, d, nd * sizeof *d);
            npend += nd;
        }
        free(d);
    }
    free(js);
    if (locked) tx_unlock();
    r->npend = r->pend ? tx_sort_unique(r->pend, npend) : 0;
    /* first block that may hold values ≥ lo */
    size_t a = 0, b = r->corrupt ? 0 : (size_t)r->nblocks;
    while (a < b) {
        size_t mid = a + (b - a) / 2;
        if (r->idx[mid].last < lo) a = mid + 1; else b = mid;
    }
    r->bi = a;
}

/* Next tested number in the range; false when exhausted. */
TX_FN int tx_reader_next(tx_reader_t *r, tx_u128 *out)
{
    for (;;) {
        /* next value of tested.bin ≥ lo */
        tx_u128 bv = 0;
        int hb = 0;
        while (!r->corrupt && r->fd >= 0) {
            if (r->bj < r->bn) {
                if (r->blk[r->bj] < r->lo) { r->bj++; continue; }
                bv = r->blk[r->bj];
                hb = bv < r->hi;
                break;
            }
            if (r->bi >= r->nblocks || r->idx[r->bi].first >= r->hi) break;
            tx_block_load(r, r->bi++);
        }
        int hp = r->pj < r->npend;
        tx_u128 pv = hp ? r->pend[r->pj] : 0;
        if (!hb && !hp) return 0;
        tx_u128 v;
        if (hb && (!hp || bv <= pv)) { v = bv; r->bj++; if (hp && pv == bv) r->pj++; }
        else { v = pv; r->pj++; }
        if (r->have_last && v == r->last) continue;
        r->have_last = 1;
        r->last = v;
        *out = v;
        return 1;
    }
}

TX_FN void tx_reader_close(tx_reader_t *r)
{
    if (r->fd >= 0) close(r->fd);
    free(r->idx);
    free(r->blk);
    free(r->pend);
    memset(r, 0, sizeof *r);
    r->fd = -1;
}

/* ------------------------------------------------------------------------- */
/* tested.bin: writing (merging the journal)                                 */

typedef struct {
    int fd;
    uint64_t off, count;
    tx_idx_t *idx;
    size_t nidx, capidx;
    unsigned char *buf;           /* block being built */
    size_t blen;
    uint32_t bcount;
    tx_u128 bfirst, blast, prev;
    int ok;
} tx_writer_t;

TX_FN void tx_w_flush_block(tx_writer_t *w)
{
    if (!w->bcount || !w->ok) return;
    if (w->nidx == w->capidx) {
        w->capidx = w->capidx ? 2 * w->capidx : 1024;
        tx_idx_t *ni = realloc(w->idx, w->capidx * sizeof *ni);
        if (!ni) { w->ok = 0; return; }
        w->idx = ni;
    }
    tx_idx_t *e = &w->idx[w->nidx++];
    e->first = w->bfirst;
    e->last = w->blast;
    e->off = w->off;
    e->size = (uint32_t)w->blen;
    e->count = w->bcount;
    e->crc = tx_crc32(w->buf, w->blen, 0);
    if (!tx_write_all(w->fd, w->buf, w->blen)) w->ok = 0;
    w->off += w->blen;
    w->blen = 0;
    w->bcount = 0;
}

TX_FN void tx_w_add(tx_writer_t *w, tx_u128 v)
{
    if (w->count && v <= w->prev) return;         /* duplicate (or out of order: impossible) */
    if (w->bcount == TX_BLOCK) tx_w_flush_block(w);
    if (!w->bcount) {
        tx_put128(w->buf, v);
        w->blen = 16;
        w->bfirst = v;
    } else {
        tx_u128 d = v - w->prev;
        while (d >= 0x80) { w->buf[w->blen++] = (unsigned char)(d | 0x80); d >>= 7; }
        w->buf[w->blen++] = (unsigned char)d;
    }
    w->bcount++;
    w->blast = w->prev = v;
    w->count++;
}

/* Merges the journal (and the legacy tested.txt) into tested.bin, rewritten in order. With
   force = 0, a large tested.bin (> TX_SMALL numbers) is only rewritten once the journal
   exceeds 1/16 of it; until then the journal is equally valid for the generator. Returns 1 on
   success. */
TX_FN int tx_compact(int force, char *err, size_t errsz, int (*wait_cb)(double))
{
    err[0] = 0;
    if (!tx_lock_wait(wait_cb, err, errsz)) return 0;
    unlink(TX.tmp);                               /* leftovers of an interrupted rewrite */
    unlink(TX.journal_tmp);
    uint64_t bad = 0;
    size_t nd = 0, nt = 0, nj = 0, capd = 0;
    int txt_exists;
    /* sources: the shared journal, mine and those of processes that already died */
    tx_jf_t *js = tx_journals(0, &nj);
    tx_u128 *d = NULL;
    int journal_exists = 0;
    for (size_t i = 0; i < nj; i++) {
        if (js[i].state == 3) continue;          /* of another live process: it handles it */
        size_t k = 0;
        tx_u128 *v = tx_journal_read(js[i].path, 0, TX_LIMIT, &k, &bad);
        struct stat jst;
        if (stat(js[i].path, &jst) == 0 && jst.st_size > 0) journal_exists = 1;
        if (k) {
            if (nd + k > capd) {
                capd = (nd + k) * 2;
                tx_u128 *nv = realloc(d, capd * sizeof *nv);
                if (!nv) { free(v); continue; }
                d = nv;
            }
            memcpy(d + nd, v, k * sizeof *v);
            nd += k;
        }
        free(v);
    }
    tx_u128 *t = tx_txt_read(0, TX_LIMIT, &nt, &bad, &txt_exists);
    tx_reader_t r;
    memset(&r, 0, sizeof r);
    r.fd = -1;
    r.blk = malloc(TX_BLOCK * sizeof *r.blk);
    tx_bin_open(&r);
    int ok = 1;
    int rewrite = !r.corrupt && (nd || nt || txt_exists)
                  && (force || txt_exists || r.count < TX_SMALL || (uint64_t)(nd + nt) * 16 >= r.count);
    int consumed = 0;                             /* the journals are already in their final place */
    if (r.corrupt) {
        snprintf(err, errsz, "%s is corrupt; nothing was changed, and the numbers remain in the journals", TX.bin);
        ok = 0;
    } else if (!nd && !nt && !txt_exists) {
        consumed = journal_exists;                /* only invalid records: safe to clean up */
    } else if (!rewrite) {
        /* large tested.bin and small journals: gather everything into the shared journal
           (the rewrite is deferred until it grows) */
        size_t np = tx_sort_unique(d, nd);
        unsigned char *buf = malloc(np * TX_REC + 1);
        int e = 0;
        if (buf) {
            for (size_t i = 0; i < np; i++) tx_encode_rec(buf + i * TX_REC, d[i]);
            /* the shared journal is rewritten (temporary + rename) without duplicates */
            int fd = open(TX.journal_tmp, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
            int wok = fd >= 0 && tx_write_all(fd, buf, np * TX_REC) && fsync(fd) == 0;
            if (!wok) e = errno;
            if (fd >= 0) close(fd);
            if (wok && rename(TX.journal_tmp, TX.journal) == 0) {
                for (size_t i = 0; i < nj; i++)
                    if (js[i].state == 1 || js[i].state == 2) unlink(js[i].path);
            } else {
                if (!e) e = errno;
                unlink(TX.journal_tmp);
                ok = 0;
                snprintf(err, errsz, "failed to write %s (%s)", TX.journal, strerror(e));
            }
            free(buf);
        } else {
            ok = 0;
            snprintf(err, errsz, "out of memory");
        }
    } else {
        tx_u128 *p = malloc((nd + nt + 1) * sizeof *p);
        tx_writer_t w;
        memset(&w, 0, sizeof w);
        w.buf = malloc(16 + 19 * TX_BLOCK);
        w.ok = p && w.buf && r.blk;
        size_t np = 0;
        if (p) {
            if (nd) memcpy(p, d, nd * sizeof *d);
            if (nt) memcpy(p + nd, t, nt * sizeof *t);
            np = tx_sort_unique(p, nd + nt);
        }
        w.fd = w.ok ? open(TX.tmp, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644) : -1;
        unsigned char h[TX_HDR];
        memset(h, 0, sizeof h);
        if (w.fd < 0 || !tx_write_all(w.fd, h, TX_HDR)) w.ok = 0;
        w.off = TX_HDR;
        /* merges the old file (block by block) with the new numbers */
        size_t pi = 0;
        for (uint64_t bi = 0; w.ok && bi < r.nblocks; bi++) {
            if (!tx_block_load(&r, bi)) {
                snprintf(err, errsz, "block %llu of %s is corrupt; nothing was changed",
                         (unsigned long long)bi, TX.bin);
                w.ok = 0;
                break;
            }
            for (size_t j = 0; j < r.bn; j++) {
                while (pi < np && p[pi] < r.blk[j]) tx_w_add(&w, p[pi++]);
                tx_w_add(&w, r.blk[j]);
            }
        }
        while (w.ok && pi < np) tx_w_add(&w, p[pi++]);
        tx_w_flush_block(&w);
        if (w.ok) {
            size_t ib = w.nidx * TX_IDX;
            unsigned char *raw = calloc(ib ? ib : 1, 1);
            if (!raw) w.ok = 0;
            for (size_t i = 0; w.ok && i < w.nidx; i++) {
                unsigned char *q = raw + i * TX_IDX;
                tx_put128(q, w.idx[i].first);
                tx_put128(q + 16, w.idx[i].last);
                tx_put64(q + 32, w.idx[i].off);
                tx_put32(q + 40, w.idx[i].size);
                tx_put32(q + 44, w.idx[i].count);
                tx_put32(q + 48, w.idx[i].crc);
            }
            if (w.ok && !tx_write_all(w.fd, raw, ib)) w.ok = 0;
            memcpy(h, TX_MAGIC, 8);
            tx_put64(h + 8, w.count);
            tx_put64(h + 16, w.nidx);
            tx_put64(h + 24, w.off);
            tx_put32(h + 32, tx_crc32(raw, ib, 0));
            tx_put32(h + 36, TX_BLOCK);
            tx_put32(h + 60, tx_crc32(h, 60, 0));
            free(raw);
            if (w.ok && (pwrite(w.fd, h, TX_HDR, 0) != TX_HDR || fsync(w.fd) != 0)) w.ok = 0;
        }
        if (w.fd >= 0) close(w.fd);
        if (w.ok && rename(TX.tmp, TX.bin) == 0) {
            int dfd = open(TX.dir, O_RDONLY | O_CLOEXEC);
            if (dfd >= 0) { fsync(dfd); close(dfd); }
            /* only now, with the new tested.bin in place, are the journals and the .txt removed */
            consumed = 1;
            if (txt_exists && rename(TX.txt, TX.txt_old) != 0) { /* retried next time: duplicates vanish */ }
        } else {
            if (!err[0]) snprintf(err, errsz, "failed to write %s (%s)", TX.tmp, strerror(errno));
            unlink(TX.tmp);
            ok = 0;
        }
        free(p);
        free(w.buf);
        free(w.idx);
    }
    if (consumed) {
        for (size_t i = 0; i < nj; i++) {
            if (js[i].state == 0) { if (truncate(js[i].path, 0) != 0) { /* kept: duplicates vanish */ } }
            else if (js[i].state != 3) unlink(js[i].path);
        }
    }
    tx_unlock();
    if (r.fd >= 0) close(r.fd);
    free(r.idx);
    free(r.blk);
    free(d);
    free(t);
    free(js);
    return ok;
}

/* ------------------------------------------------------------------------- */
/* data.csv                                                                  */

#define TX_DATA_HEADER "n,k,odd_k,j_minus_k,odd_j_minus_k,exponent_j,m,x_m,odd_m\n"
#define TX_DATA_COLS   9

/* Appends data lines (text) to this process's data journal, without lock. */
TX_FN int tx_data_append(const char *buf, size_t len, int *err_out)
{
    return tx_journal_append(TX.my_data, buf, len, 0, err_out);
}

/* Valid data line? Returns n (or 0). Integer fields; only j_minus_k may be negative. */
TX_FN tx_u128 tx_data_key(const char *l, size_t len)
{
    size_t col = 0, digits = 0;
    tx_u128 n = 0;
    for (size_t i = 0; i <= len; i++) {
        char c = i < len ? l[i] : ',';
        if (c == ',') {
            if (!digits) return 0;
            col++;
            digits = 0;
            continue;
        }
        if (c == '-' && col == 3 && digits == 0 && (i == 0 || l[i - 1] == ',')) continue;
        if (c < '0' || c > '9') return 0;
        if (col == 0) {
            if (n > (~(tx_u128)0 - 9) / 10) return 0;
            n = n * 10 + (tx_u128)(c - '0');
        }
        digits++;
    }
    return col == TX_DATA_COLS && n >= TX_MIN && n < TX_LIMIT ? n : 0;
}

typedef struct { tx_u128 n; char *line; size_t len; } tx_row_t;

TX_FN int tx_row_cmp(const void *a, const void *b)
{
    tx_u128 x = ((const tx_row_t *)a)->n, y = ((const tx_row_t *)b)->n;
    return x < y ? -1 : x > y;
}

/* Merges data.journal.csv into data.csv, ordered by n (a repeated n is kept only once). */
TX_FN int tx_data_compact(char *err, size_t errsz, int (*wait_cb)(double))
{
    err[0] = 0;
    if (!tx_lock_wait(wait_cb, err, errsz)) return 0;
    unlink(TX.data_tmp);                          /* leftover of an interrupted rewrite */
    int ok = 1;
    tx_row_t *rows = NULL;
    size_t nr = 0, cap = 0, nj = 0;
    char *line = NULL;
    size_t lcap = 0;
    ssize_t len;
    tx_jf_t *js = tx_journals(1, &nj);
    for (size_t ji = 0; ok && ji < nj; ji++) {
        if (js[ji].state == 3) continue;          /* of another live process */
        FILE *fd = fopen(js[ji].path, "r");
        if (!fd) continue;
        while ((len = getline(&line, &lcap, fd)) > 0) {
            if (line[len - 1] != '\n') break;                 /* truncated line */
            tx_u128 n = tx_data_key(line, (size_t)len - 1);
            if (!n) continue;
            if (nr == cap) {
                cap = cap ? 2 * cap : 1024;
                tx_row_t *nv = realloc(rows, cap * sizeof *nv);
                if (!nv) { ok = 0; break; }
                rows = nv;
            }
            rows[nr].n = n;
            rows[nr].len = (size_t)len;
            rows[nr].line = malloc((size_t)len);
            if (!rows[nr].line) { ok = 0; break; }
            memcpy(rows[nr].line, line, (size_t)len);
            nr++;
        }
        fclose(fd);
    }
    int consumed = ok && !nr;                     /* journals empty or holding only garbage */
    if (ok && nr) {
        qsort(rows, nr, sizeof *rows, tx_row_cmp);           /* qsort is not stable: irrelevant here */
        FILE *in = fopen(TX.data, "r"), *out = fopen(TX.data_tmp, "w");
        if (!out) ok = 0;
        if (ok) fputs(TX_DATA_HEADER, out);
        size_t ri = 0;
        tx_u128 last = 0;
        int have_last = 0;
        while (ok && in && (len = getline(&line, &lcap, in)) > 0) {
            if (line[len - 1] != '\n') break;
            tx_u128 n = tx_data_key(line, (size_t)len - 1);
            if (!n) continue;                                 /* header or invalid line */
            while (ri < nr && rows[ri].n < n) {
                if (!have_last || rows[ri].n != last) fwrite(rows[ri].line, 1, rows[ri].len, out);
                last = rows[ri].n; have_last = 1; ri++;
            }
            if (!have_last || n != last) fwrite(line, 1, (size_t)len, out);
            last = n; have_last = 1;
        }
        while (ok && ri < nr) {
            if (!have_last || rows[ri].n != last) fwrite(rows[ri].line, 1, rows[ri].len, out);
            last = rows[ri].n; have_last = 1; ri++;
        }
        if (in) fclose(in);
        if (out) {
            if (fflush(out) != 0 || ferror(out) || fsync(fileno(out)) != 0) ok = 0;
            if (fclose(out) != 0) ok = 0;
        }
        if (ok && rename(TX.data_tmp, TX.data) == 0) {
            consumed = 1;
        } else {
            snprintf(err, errsz, "failed to write %s (%s)", TX.data, strerror(errno));
            unlink(TX.data_tmp);
            ok = 0;
        }
    }
    if (consumed)
        for (size_t i = 0; i < nj; i++) {
            if (js[i].state == 0) { if (truncate(js[i].path, 0) != 0) { /* kept: duplicates vanish */ } }
            else if (js[i].state != 3) unlink(js[i].path);
        }
    for (size_t i = 0; i < nr; i++) free(rows[i].line);
    free(rows);
    free(line);
    free(js);
    tx_unlock();
    return ok;
}

#endif
