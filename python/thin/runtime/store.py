"""The store.

What was computed, filed under an address made from what the computation
read.

It knows nothing of what it holds, uses only the standard library, and runs
on Linux. Its file format and lock protocol are stable: multiple processes and thin
versions may share a store directory.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import enum
import errno
import fcntl
import hashlib
import json
import logging
import os
import re
import secrets
import socket
import sys
import time

__all__ = [
    "HEAD",
    "LOCK_PREFIX",
    "MAGIC",
    "Miss",
    "address",
    "content_digest",
    "digest",
    "encode",
    "entry",
    "evict",
    "get",
    "lookup",
    "main",
    "publish",
    "report",
]

MAGIC = b"thin.store 1\n"
"""The first bytes of every entry."""

HEAD = 65536
"""The bytes a lookup reads first: magic and header must fit in them."""

LOCK_PREFIX = "thin-store-"
"""The per-user lock directory's prefix, under /dev/shm."""

log = logging.getLogger("thin.store")
"""The store's logger."""

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_ADDRESS = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)/([0-9a-f]{64})\Z")


# --- Digests and addresses ----------------------------------------


def _encode(v, out: list) -> None:
    if isinstance(v, tuple):
        out += [b"t", len(v).to_bytes(8, "little")]
        for m in v:
            _encode(m, out)
    elif isinstance(v, str):
        b = str.encode(v, "utf-8")
        out += [b"s", len(b).to_bytes(8, "little"), b]
    elif isinstance(v, bytes):
        out += [b"b", len(v).to_bytes(8, "little"), v]
    elif isinstance(v, int) and not isinstance(v, bool):
        t = int.__repr__(v).encode()
        out += [b"i", len(t).to_bytes(8, "little"), t]
    else:
        raise TypeError(
            f"a digest is made of str, bytes, int and tuples of them, not {type(v).__name__}: {v!r}"
        )


def encode(*values) -> bytes:
    """The typed, length-prefixed encoding of the tuple of values."""
    out: list = []
    _encode(tuple(values), out)
    return b"".join(out)


def digest(*values) -> str:
    """The sha256 hex of ``encode(*values)``."""
    return hashlib.sha256(encode(*values)).hexdigest()


def address(name, *values) -> str:
    """``<name>/<digest(name, *values)>``: the one way to make an address."""
    if not isinstance(name, str) or not _NAME.match(name) or name == "tmp":
        raise ValueError(f"an address's name is an identifier other than 'tmp', not {name!r}")
    return f"{name}/{digest(name, *values)}"


def _checked(addr) -> tuple[str, str]:
    m = _ADDRESS.match(addr) if isinstance(addr, str) else None
    if m is None or m.group(1) == "tmp":
        raise ValueError(
            f"not an address (want '<name>/<64 hex digits>', as address() makes): {addr!r}"
        )
    return m.group(1), m.group(2)


def path_of(root, addr) -> str:
    """The entry file of an address: ``<root>/<name>/<hh>/<hex>``."""
    name, hexd = _checked(addr)
    return os.path.join(os.path.expanduser(root), name, hexd[:2], hexd)


# --- Entries -------------------------------------------------------------


def _compact(x) -> bytes:
    return json.dumps(x, separators=(",", ":")).encode()


def _members_list(members) -> list:
    ok = isinstance(members, dict) and all(
        isinstance(k, str) and isinstance(v, bytes) for k, v in members.items()
    )
    if not ok:
        raise TypeError(f"what is stored is a dict of str to bytes, not {type(members).__name__}")
    return [[k, len(v), hashlib.sha256(v).hexdigest()] for k, v in members.items()]


def entry(addr, members) -> bytes:
    """The entry file of members at an address."""
    _checked(addr)
    listed = _members_list(members)
    header = _compact({"address": addr, "members": listed}) + b"\n"
    if len(MAGIC) + len(header) > HEAD:
        raise ValueError(
            f"{addr}: a header of {len(listed)} members takes {len(header)} bytes, past the "
            f"{HEAD - len(MAGIC)} a lookup reads"
        )
    return MAGIC + header + b"".join(members.values())


def content_digest(members) -> str:
    """The sha256 of the compact JSON of an entry's members list."""
    return hashlib.sha256(_compact(_members_list(members))).hexdigest()


# --- Counters -------------------------------------------------------------

_COUNTS: dict[str, float] = {}


def _count(addr: str, what: str, n: float = 1) -> None:
    key = f"{addr.split('/')[0]}.{what}"
    _COUNTS[key] = _COUNTS.get(key, 0) + n


def _total(what: str, n: float) -> None:
    _COUNTS[what] = _COUNTS.get(what, 0) + n


def report() -> dict:
    """A copy of this process's counters."""
    return dict(_COUNTS)


# --- Lookup ---------------------------------------------------------------


class Miss(enum.Enum):
    """Why a lookup found no whole entry."""

    ABSENT = "absent"
    CORRUPT = "corrupt"
    FOREIGN = "foreign"
    VANISHED = "vanished"


_GONE = (errno.ENOENT, errno.ESTALE)


def _header(head: bytes):
    """(address, members, data offset), or None when not whole."""
    if not head.startswith(MAGIC):
        return None
    end = head.find(b"\n", len(MAGIC))
    if end < 0:
        return None
    try:
        h = json.loads(head[len(MAGIC) : end])
    except ValueError:
        return None
    if (
        not isinstance(h, dict)
        or not isinstance(h.get("address"), str)
        or not isinstance(h.get("members"), list)
    ):
        return None
    for m in h["members"]:
        shaped = (
            isinstance(m, list) and len(m) == 3 and isinstance(m[0], str) and isinstance(m[2], str)
        )
        if not shaped or type(m[1]) is not int or m[1] < 0:
            return None
    return h["address"], h["members"], end + 1


def _read(fd: int, head: bytes, at: int, n: int) -> bytes:
    if at + n <= len(head):
        return head[at : at + n]
    return os.pread(fd, n, at)


def _classify(fd: int, addr: str, names):
    head = os.pread(fd, HEAD, 0)
    _total("bytes read", len(head))
    parsed = _header(head)
    if parsed is None:
        return Miss.CORRUPT
    theirs, listed, at = parsed
    size = os.fstat(fd).st_size
    if size != at + sum(m[1] for m in listed):
        return Miss.CORRUPT
    if theirs != addr:
        return Miss.FOREIGN
    where, k = {}, at
    for name, n, sha in listed:
        where[name] = (k, n, sha)
        k += n
    chosen = list(where) if names is None else list(names)
    out = {}
    for name in chosen:
        if name not in where:
            return Miss.FOREIGN
        k, n, sha = where[name]
        data = _read(fd, head, k, n)
        if k + n > len(head):
            _total("bytes read", n)
        if hashlib.sha256(data).hexdigest() != sha:
            return Miss.CORRUPT
        out[name] = data
    return out


def lookup(root, addr, names=None):
    """The entry's members (the chosen ones), or a Miss; writes nothing."""
    path = path_of(root, addr)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOATIME if _noatime() else os.O_RDONLY)
    except FileNotFoundError:
        return Miss.ABSENT
    except OSError as e:
        if e.errno == errno.ESTALE:
            return Miss.VANISHED
        if e.errno == errno.EPERM and _noatime():
            fd = os.open(path, os.O_RDONLY)
        else:
            raise
    try:
        return _classify(fd, addr, names)
    except OSError as e:
        if e.errno in _GONE:
            return Miss.VANISHED
        raise
    finally:
        os.close(fd)


def _noatime() -> bool:
    return hasattr(os, "O_NOATIME")


def _looked(root, addr, names):
    found = lookup(root, addr, names)
    if isinstance(found, Miss):
        _count(addr, f"miss.{found.value}")
        if found is not Miss.ABSENT:
            log.warning("not used: %s", path_of(root, addr))
    return found


# --- Publishing -------------------------------------------------------------


def _short(members) -> str:
    return content_digest(members)[:16]


def publish(root, addr, members) -> str:
    """Files the entry by a hard link; returns ``published``, ``raced equal`` or ``not stored``."""
    data = entry(addr, members)
    root = os.path.expanduser(root)
    final = path_of(root, addr)
    tmpdir = os.path.join(root, "tmp")
    tmp = os.path.join(tmpdir, f"{socket.gethostname()}-{os.getpid()}-{secrets.token_hex(4)}")
    try:
        os.makedirs(tmpdir, exist_ok=True)
        os.makedirs(os.path.dirname(final), exist_ok=True)
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view) :]
            os.fsync(fd)
        finally:
            os.close(fd)
        _total("bytes written", len(data))
        try:
            os.link(tmp, final)
            outcome = "published"
        except FileExistsError:
            outcome = _raced(root, addr, members)
    except OSError as e:
        log.warning("%s not stored: %s", addr, errno.errorcode.get(e.errno, str(e.errno)))
        outcome = "not stored"
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
    _count(addr, outcome)
    return outcome


def _raced(root, addr, members) -> str:
    theirs = lookup(root, addr)
    if isinstance(theirs, Miss):
        log.warning(
            "a %s file holds the name (%s); evict removes it", theirs.value, path_of(root, addr)
        )
        return "not stored"
    if content_digest(theirs) == content_digest(members):
        return "raced equal"
    _count(addr, "collision")
    from thin.runtime.errors import Collision

    raise Collision(
        f"{addr} holds other bytes (theirs {_short(theirs)}, these {_short(members)}) in {root}: "
        "a producer "
        "that is not deterministic, or an address missing something its computation read"
    )


# --- Locks ------------------------------------------------------------------


def _lock_path(root, addr) -> str:
    folder = f"/dev/shm/{LOCK_PREFIX}{os.getuid()}"
    os.makedirs(folder, mode=0o700, exist_ok=True)
    real = os.path.realpath(os.path.expanduser(root))
    rid = hashlib.sha256(real.encode()).hexdigest()[:16]
    return os.path.join(folder, f"{rid}-{_checked(addr)[1]}")


class _Lock:
    """The host lock of an address: a flocked file a holder unlinks before releasing."""

    def __init__(self, root, addr):
        self.path = _lock_path(root, addr)
        self.fd = None
        self.contended = False

    def take(self, wait: float) -> bool:
        deadline, pause = time.monotonic() + wait, 0.002
        while True:
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                self.contended = True
                if time.monotonic() >= deadline:
                    return False
                time.sleep(pause)
                pause = min(pause * 2, 0.05)
                continue
            try:
                same = os.fstat(fd).st_ino == os.stat(self.path).st_ino
            except FileNotFoundError:
                same = False
            if same:
                self.fd = fd
                return True
            os.close(fd)

    def release(self) -> None:
        if self.fd is None:
            return
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.path)
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)
        self.fd = None


# --- get ----------------------------------------------------------------------


def _changed(sources) -> list[str]:
    from thin.runtime.packages import _sha

    out = []
    for path, sha in sources.items():
        try:
            now = _sha(path)
        except OSError:
            out.append(f"{path} (gone)")
            continue
        if now != sha:
            out.append(f"{path} (its bytes now differ)")
    return out


def get(root, addr, compute, names=None, *, wait, sources):
    """The members at the address, computed at most once per host at a time."""
    start = time.monotonic()
    first = lookup(root, addr, names)
    if not isinstance(first, Miss):
        _count(addr, "hit")
        _total("store seconds", time.monotonic() - start)
        return first
    lock = _Lock(root, addr)
    waited = time.monotonic()
    took = lock.take(wait)
    _total("waiting seconds", time.monotonic() - waited)
    try:
        if took:
            again = _looked(root, addr, names)
            if not isinstance(again, Miss):
                _count(addr, "waited")
                return again
            if lock.contended:
                _count(addr, "holder left nothing")
                log.warning("%s: the holder left nothing", addr)
        else:
            _count(addr, "waited too long")
            log.warning("a holder past %s s: computed here as well", wait)
            if isinstance(first, Miss):
                _count(addr, f"miss.{first.value}")
        members = compute()
        _count(addr, "computed")
        chosen = list(members) if names is None else list(names)
        lacking = [n for n in chosen if n not in members]
        if lacking:
            raise KeyError(f"{addr}: compute() returned no {lacking} (it returned {list(members)})")
        changed = _changed(sources)
        if changed:
            more = f" and {len(changed) - 3} more" if len(changed) > 3 else ""
            log.warning("its files changed: %s%s", ", ".join(changed[:3]), more)
            _count(addr, "not stored")
        else:
            publish(root, addr, members)
        return {n: members[n] for n in chosen}
    finally:
        lock.release()
        _total("store seconds", time.monotonic() - start)


# --- Eviction and the command line ------------------------------------------------


def _whole_header(path) -> bool:
    with open(path, "rb") as f:
        head = f.read(HEAD)
        size = os.fstat(f.fileno()).st_size
    parsed = _header(head)
    return parsed is not None and size == parsed[2] + sum(m[1] for m in parsed[1])


def evict(root, before, digests=False) -> int:
    """Removes entries older than ``before`` or not whole; returns how many."""
    root = os.path.expanduser(root)
    removed = 0
    names = sorted(n for n in _listdir(root) if _NAME.match(n) and n != "tmp")
    for name in names:
        for sub in sorted(_listdir(os.path.join(root, name))):
            folder = os.path.join(root, name, sub)
            for hexd in sorted(_listdir(folder)):
                if not re.fullmatch(r"[0-9a-f]{64}", hexd):
                    continue
                removed += _evict_one(
                    root, f"{name}/{hexd}", os.path.join(folder, hexd), before, digests
                )
    tmp = os.path.join(root, "tmp")
    for leftover in _listdir(tmp):
        p = os.path.join(tmp, leftover)
        try:
            if os.stat(p).st_mtime < time.time() - 3600:
                os.unlink(p)
        except OSError:
            pass
    return removed


def _evict_one(root, addr, path, before, digests) -> int:
    try:
        old = os.stat(path).st_mtime < before
        broken = not _whole_header(path) or _foreign(path, addr)
        if digests and not old and not broken:
            broken = isinstance(_looked(root, addr, None), Miss)
        if old or broken:
            os.unlink(path)
    except FileNotFoundError:
        return 1
    except IsADirectoryError:
        return 0
    return int(old or broken)


def _foreign(path, addr) -> bool:
    with open(path, "rb") as f:
        parsed = _header(f.read(HEAD))
    return parsed is not None and parsed[0] != addr


def _listdir(path) -> list[str]:
    try:
        return os.listdir(path)
    except OSError:
        return []


def _date(text: str) -> float:
    return datetime.datetime.fromisoformat(text).timestamp()


def main(argv=None) -> int:
    """``python -m thin.store evict ROOT --before DATE [--digests]``."""
    p = argparse.ArgumentParser(prog="python -m thin.store")
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("evict", help="remove entries older than DATE or not whole")
    e.add_argument("root")
    e.add_argument("--before", required=True, type=_date, help="ISO 8601; local time unless zoned")
    e.add_argument("--digests", action="store_true", help="check every member's digest too")
    a = p.parse_args(argv)
    print(f"{evict(a.root, a.before, a.digests)} removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
