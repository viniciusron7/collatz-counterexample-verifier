"""
Independent (Python) reader of the Tested/ directory format (see tested.h), used by the tests.

    tested.bin       64-byte header, blocks (1st value in 16 bytes + LEB128 differences) and
                     a 56-byte index entry per block; everything with CRC-32
    tested.journal   20-byte records: n (16 bytes little-endian) + CRC-32 (the shared
                     journal); tested.journal.<pid>: same format, one per process
"""
import os
import struct
import zlib

MAGIC = b"CLZTST01"
MIN, LIMIT = 2 ** 71, 2 ** 127
JOURNAL = "tested.journal"


def crc(b, seed=0):
    return zlib.crc32(b, seed) & 0xFFFFFFFF


def read_bin(path):
    """List of the numbers in tested.bin (raises ValueError if any check fails)."""
    data = open(path, "rb").read()
    h = data[:64]
    if h[:8] != MAGIC or struct.unpack_from("<I", h, 60)[0] != crc(h[:60]):
        raise ValueError("header")
    count, nblocks, ioff, icrc, bmax = struct.unpack_from("<QQQII", h, 8)
    idx = data[ioff:ioff + 56 * nblocks]
    if ioff + 56 * nblocks != len(data) or crc(idx) != icrc:
        raise ValueError("index")
    out = []
    for i in range(nblocks):
        e = idx[56 * i:56 * (i + 1)]
        first = int.from_bytes(e[0:16], "little")
        last = int.from_bytes(e[16:32], "little")
        off, size, cnt, bcrc = struct.unpack_from("<QIII", e, 32)
        blk = data[off:off + size]
        if crc(blk) != bcrc or cnt > bmax:
            raise ValueError(f"block {i}")
        v = int.from_bytes(blk[:16], "little")
        vals = [v]
        p = 16
        while len(vals) < cnt:
            d = sh = 0
            while True:
                c = blk[p]
                p += 1
                d |= (c & 0x7F) << sh
                sh += 7
                if not c & 0x80:
                    break
            v += d
            vals.append(v)
        if p != size or vals[0] != first or vals[-1] != last:
            raise ValueError(f"block {i} (content)")
        out += vals
    if len(out) != count or any(a >= b for a, b in zip(out, out[1:])):
        raise ValueError("count or order")
    return out


def read_journal(path):
    """Intact numbers of the journal, in the order they were written."""
    if not os.path.exists(path):
        return []
    data = open(path, "rb").read()
    out = []
    for i in range(0, len(data) - len(data) % 20, 20):
        r = data[i:i + 20]
        if struct.unpack_from("<I", r, 16)[0] == crc(r[:16], 0x5EED7E57):
            v = int.from_bytes(r[:16], "little")
            if MIN <= v < LIMIT:
                out.append(v)
    return out


def record(v):
    b = v.to_bytes(16, "little")
    return b + struct.pack("<I", crc(b, 0x5EED7E57))


def journals(d):
    """Paths of every number journal in directory d (shared and per process)."""
    if not os.path.isdir(d):
        return []
    out = []
    for nm in sorted(os.listdir(d)):
        if nm == JOURNAL or (nm.startswith(JOURNAL + ".") and nm[len(JOURNAL) + 1:].isdigit()):
            out.append(os.path.join(d, nm))
    return out


def read_journals(d):
    """Intact numbers of every journal in directory d."""
    out = []
    for f in journals(d):
        out += read_journal(f)
    return out


def stored(d):
    """Set of everything stored in directory d (tested.bin + journals)."""
    b = os.path.join(d, "tested.bin")
    s = set(read_bin(b)) if os.path.exists(b) else set()
    return s | set(read_journals(d))
