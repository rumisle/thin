"""The files an address is made from: packages, their digest and the trace test."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
import py_compile
import re
import stat
import subprocess
import sys
import sysconfig
from dataclasses import dataclass, field

__all__ = [
    "Dist",
    "files_of",
    "hashes",
    "interpreter",
    "outside",
    "packages",
    "packages_digest",
    "packages_for",
    "read_by",
]

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _real(p) -> str:
    return os.path.realpath(p)


def _inside(path: str, folder: str) -> bool:
    return path == folder or path.startswith(folder.rstrip(os.sep) + os.sep)


@dataclass
class Dist:
    """A distribution as the address reads it."""

    name: str
    version: str
    requires: list[str]
    tops: list[str]
    editable: bool
    folders: list[str]
    claimed: dict = field(default_factory=dict)


def _requirements(d) -> list[str]:
    out = []
    for r in d.requires or ():
        head, _, marker = r.partition(";")
        if "extra" in marker:
            continue
        m = re.match(r"[A-Za-z0-9._-]*", head.strip())
        if m and m.group(0):
            out.append(_norm(m.group(0)))
    return out


def _record(d) -> list[str]:
    """The paths a distribution's RECORD lists (CSV: a quoted path is read as one)."""
    text = d.read_text("RECORD") or ""
    return [row[0] for row in csv.reader(text.splitlines()) if row]


def _pth_folders(d, record) -> list[str]:
    out = []
    for f in record:
        if f.endswith(".pth"):
            try:
                text = d.locate_file(f).read_text()
            except OSError:
                continue
            for line in text.splitlines():
                line = line.strip()
                if line and not line.startswith(("#", "import ")) and os.path.isdir(line):
                    out.append(_real(line))
    return out


def _editable(d) -> tuple[bool, str | None]:
    text = d.read_text("direct_url.json")
    if not text:
        return False, None
    try:
        u = json.loads(text)
    except ValueError:
        return False, None
    url = u.get("url", "")
    folder = _real(url[len("file://") :]) if url.startswith("file://") else None
    return bool(u.get("dir_info", {}).get("editable")), folder


def _tops(d, record, pth: list[str]) -> list[str]:
    listed = d.read_text("top_level.txt")
    if listed:
        return [t.strip() for t in listed.splitlines() if t.strip()]
    tops = set()
    for f in record:
        first = f.replace("\\", "/").split("/")[0]
        if first.endswith(".py") and _IDENT.match(first[:-3]):
            tops.add(first[:-3])
        elif "/" in f and _IDENT.match(first):
            tops.add(first)
    for folder in pth:
        for entry in os.listdir(folder):
            p = os.path.join(folder, entry)
            if (
                os.path.isdir(p)
                and _IDENT.match(entry)
                and os.path.exists(os.path.join(p, "__init__.py"))
            ):
                tops.add(entry)
            elif entry.endswith(".py") and _IDENT.match(entry[:-3]):
                tops.add(entry[:-3])
    return sorted(tops)


_DISTS: dict[str, Dist] | None = None


def distributions() -> dict[str, Dist]:
    """The distributions on the interpreter's path, the first of each normalized name."""
    global _DISTS
    if _DISTS is None:
        found: dict[str, Dist] = {}
        for d in importlib.metadata.distributions():
            name = _norm(d.metadata["Name"] or "")
            if not name or name in found:
                continue
            record = _record(d)
            pth = _pth_folders(d, record)
            editable, folder = _editable(d)
            folders = pth + ([folder] if folder else []) + [_real(d.locate_file(""))]
            tops = _tops(d, record, pth)
            found[name] = Dist(name, d.version, _requirements(d), tops, editable, folders)
        _DISTS = found
    return _DISTS


def _portions(top: str) -> list[str]:
    """The import system's locations of a top (a package's folders, or a module's file).

    An absolute top is a loose folder (its importable entries) or a file.
    """
    if os.path.isabs(top):
        return _importable(top) if os.path.isdir(top) else [top]
    try:
        spec = importlib.util.find_spec(top)
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        return []
    if spec.submodule_search_locations:
        return [_real(p) for p in spec.submodule_search_locations]
    return [_real(spec.origin)] if spec.origin and os.path.isfile(spec.origin) else []


def _portions_in(d: Dist, top: str) -> list[str]:
    return [p for p in _portions(top) if any(_inside(p, f) for f in d.folders)]


def _owners(path: str) -> list[Dist]:
    out = []
    for d in distributions().values():
        if not any(_inside(path, f) for f in d.folders):
            continue
        if any(any(_inside(path, p) for p in _portions_in(d, t)) for t in d.tops):
            out.append(d)
    return out


def _follow(owners, hashed: dict, versioned: set) -> None:
    dists, seen, todo = distributions(), set(), list(owners)
    owner_names = {d.name for d in owners}
    while todo:
        d = todo.pop()
        if d.name in seen:
            continue
        seen.add(d.name)
        if d.editable or d.name in owner_names:
            for t in d.tops:
                portions = _portions_in(d, t)
                if portions:
                    hashed[t] = portions
        else:
            versioned.add((d.name, d.version))
        todo.extend(dists[r] for r in d.requires if r in dists)


def _regular(path: str) -> bool:
    """Whether a path is a regular file (after symlinks): never a socket, FIFO or device."""
    try:
        return stat.S_ISREG(os.stat(path).st_mode)
    except OSError:
        return False


def _is_package(folder: str) -> bool:
    return _IDENT.match(os.path.basename(folder)) is not None and _regular(
        os.path.join(folder, "__init__.py")
    )


def _search_root(path: str) -> str:
    """The folder a loose file is imported from: above every package that holds it.

    A script in a plain folder is imported from that folder; a module inside
    folders with ``__init__.py`` is imported from the folder above the
    outermost of them.
    """
    root = os.path.dirname(path)
    while _is_package(root) and os.path.dirname(root) != root:
        root = os.path.dirname(root)
    return root


def _module_name(entry: str) -> bool:
    """Whether a file name is one the import system loads (a source, bytecode or extension)."""
    for suffix in sorted(importlib.machinery.all_suffixes(), key=len, reverse=True):
        if entry.endswith(suffix):
            return _IDENT.match(entry[: -len(suffix)]) is not None
    return False


def _importable(root: str) -> list[str]:
    """What an import can load from a folder on the path: its modules and its packages.

    These are a loose generator's package. Anything else in the folder (data,
    folders without ``__init__.py``, sockets) cannot be imported from it, so
    it cannot change the render and is not read.
    """
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return []
    out = []
    for entry in entries:
        p = os.path.join(root, entry)
        module = _module_name(entry) and _regular(p)
        if module or (os.path.isdir(p) and _is_package(p)):
            out.append(p)
    return out


def packages_for(paths) -> tuple[tuple, tuple]:
    """(sorted hashed tops and loose folders, sorted (name, version) pairs) of these files.

    A file no distribution owns is loose: its package is what an import can
    load from the folder it is imported from (``_search_root``), that is the
    modules and packages there, and that folder is its top.
    """
    hashed: dict[str, list[str]] = {}
    versioned: set = set()
    owners: list[Dist] = []
    for path in map(_real, paths):
        mine = _owners(path)
        if not mine:
            root = _search_root(path)
            hashed[root] = _importable(root)
            if not any(_inside(path, p) for p in hashed[root]):
                # A script whose name is no module name (``my-gen.py``) is
                # still the generator's own source: a top of its own.
                hashed[path] = [path]
            mine = [
                d
                for d in distributions().values()
                if d.editable and any(_inside(path, f) for f in d.folders)
            ]
        owners.extend(mine)
    _follow(owners, hashed, versioned)
    _CLAIMED.update(hashed)
    return tuple(sorted(hashed)), tuple(sorted(versioned))


_CLAIMED: dict[str, list[str]] = {}


def packages(tops) -> tuple[tuple, tuple]:
    """``packages_for`` starting from the distributions that install the named tops."""
    hashed: dict[str, list[str]] = {}
    versioned: set = set()
    owners = []
    for t in tops:
        mine = [d for d in distributions().values() if t in d.tops]
        if not mine:
            hashed[t] = _portions(t)
        for d in mine:
            hashed[t] = _portions_in(d, t)
        owners.extend(mine)
    _follow(owners, hashed, versioned)
    _CLAIMED.update(hashed)
    return tuple(sorted(hashed)), tuple(sorted(versioned))


def _walk(portion: str):
    """The regular files of a portion, in sorted order.

    ``__pycache__`` and hidden folders hold no source; special files (a
    socket, a FIFO) have no bytes to hash and opening one could block, so
    they are skipped, as are broken links. Folder links are not followed.
    """
    if os.path.isfile(portion):
        yield portion
        return
    for folder, dirs, names in os.walk(portion):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__" and not d.startswith("."))
        for n in sorted(names):
            p = os.path.join(folder, n)
            if _regular(p):
                yield p


def _files(top: str) -> list[tuple[str, str]]:
    """(folder the paths are relative to, path) of every hashed file of a top."""
    portions = _CLAIMED.get(top) or _portions(top)
    if not portions and not os.path.isabs(top):
        raise ValueError(f"{top!r}: no such package on this Python's path")
    out = []
    for portion in portions:
        base = top if os.path.isdir(top) and os.path.isabs(top) else os.path.dirname(portion)
        out.extend((base, p) for p in _walk(portion))
    return out


def _sha(path: str, memo: dict | None = None) -> str:
    if memo is not None and path in memo:
        return memo[path]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        # In pieces: a package may hold a native library of tens of MB.
        for piece in iter(lambda: f.read(1 << 20), b""):
            h.update(piece)
    sha = h.hexdigest()
    if memo is not None:
        memo[path] = sha
    return sha


def files_of(top: str, memo: dict | None = None) -> list[tuple[str, str]]:
    """(path relative to the folder holding the portion, sha256) of every file of a top.

    A loose folder's paths are relative to the folder itself, so the same
    script hashes the same wherever it lives.
    """
    out = [(os.path.relpath(p, base), _sha(p, memo), p) for base, p in _files(top)]
    for _, _, p in out:
        if p.endswith(".py"):
            _checked_bytecode(p)
    return [(rel, sha) for rel, sha, _ in out]


def _checked_bytecode(path: str) -> None:
    """Hash-based, checked bytecode for a hashed source (PEP 552 flags 0b11)."""
    cache = importlib.util.cache_from_source(path)
    if not os.access(os.path.dirname(path), os.W_OK):
        return
    try:
        with open(cache, "rb") as f:
            head = f.read(8)
        if head[:4] == importlib.util.MAGIC_NUMBER and int.from_bytes(head[4:8], "little") == 0b11:
            return
    except OSError:
        pass
    tmp = f"{cache}.{os.getpid()}.tmp"
    try:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        py_compile.compile(
            path,
            cfile=tmp,
            doraise=True,
            invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH,
        )
        os.replace(tmp, cache)
    except (py_compile.PyCompileError, OSError, SyntaxError):
        with contextlib.suppress(OSError):
            os.unlink(tmp)


def interpreter() -> tuple[str, str]:
    """(implementation name, version x.y.z)."""
    v = sys.version_info
    return (sys.implementation.name, f"{v.major}.{v.minor}.{v.micro}")


def packages_digest(hashed, versions, memo: dict | None = None) -> str:
    """``digest((files, versions, interpreter))``; memo shares hashes."""
    from thin.runtime.store import digest

    files = tuple(sorted(f for top in hashed for f in files_of(top, memo)))
    return digest((files, tuple(tuple(v) for v in versions), interpreter()))


def hashes(hashed, memo: dict | None = None) -> dict[str, str]:
    """{real path: sha256} of every hashed file: the ``sources`` a get re-hashes."""
    return {_real(p): _sha(p, memo) for top in hashed for _, p in _files(top)}


_TRACER = r"""
import sys
code = sys.stdin.read()
opened, tracing = set(), [False]
def hook(event, args):
    if tracing[0] and event == "open" and args and isinstance(args[0], str):
        opened.add(args[0])
sys.addaudithook(hook)
start = set(sys.modules)
tracing[0] = True
try:
    exec(compile(code, "<traced>", "exec"), {"__name__": "__main__"})
finally:
    tracing[0] = False
for name, m in list(sys.modules.items()):
    if name not in start and isinstance(getattr(m, "__file__", None), str):
        opened.add(m.__file__)
sys.stdout.write("\0".join(sorted(opened)))
"""


def _stdlib(real: str) -> bool:
    """Under the stdlib or platstdlib folder, and not under a site-packages there."""
    paths = sysconfig.get_paths()
    for key in ("stdlib", "platstdlib"):
        root = _real(paths[key])
        if _inside(real, root) and "site-packages" not in os.path.relpath(real, root).split(os.sep):
            return True
    return False


def read_by(code: str) -> set[str]:
    """The files a fresh interpreter opens or imports while running ``code``.

    The audit hook is installed after the interpreter's startup (site and
    its.pth files untraced); what counts is every str path of an ``open``
    event while ``code`` runs and the file of every module it imports, as
    real paths, less non-files, ``__pycache__`` and the standard library.
    """
    from thin.runtime.env import child

    r = subprocess.run(
        [sys.executable, "-B", "-c", _TRACER],
        input=code,
        capture_output=True,
        text=True,
        env=child(),
        check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(f"the traced code failed: {r.stderr[-2000:]}")
    out = set()
    for p in filter(None, r.stdout.split("\0")):
        real = _real(p)
        if not os.path.isfile(real) or "__pycache__" in real.split(os.sep) or _stdlib(real):
            continue
        out.add(real)
    return out


def outside(code: str, hashed, versions) -> set[str]:
    """``read_by(code)`` minus the hashed packages' and versioned distributions' files.

    A versioned distribution's files are every file its RECORD lists, its
    own dist-info included.
    """
    mine = set(hashes(hashed))
    dists = distributions()
    for name, _ in versions:
        d = importlib.metadata.distribution(name) if name in dists else None
        for f in _record(d) if d else ():
            mine.add(_real(d.locate_file(f)))
    return read_by(code) - mine
