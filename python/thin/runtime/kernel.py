"""KernelDef, compile and the Kernel."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from thin.runtime.params import bind_params, declare, held_pairs

__all__ = ["FORMAT", "Kernel", "KernelDef", "cdiv", "compile", "render_address", "use_store"]

# Cache-identity encoding version, independent of the package version.
FORMAT = 2
"""The format of a stored render: members ``s`` and ``co``."""

_WAIT = 1800
"""How long a compile waits for another process's render (seconds)."""


@dataclass(frozen=True, eq=False)
class KernelDef:
    """What a generator returns."""

    symbol: str
    """The kernel's symbol."""
    args: object
    """Its argument block."""
    body: object
    """``body(b, k)``: its code, b the Builder and k the Entry of ``args``."""
    wgsize: int
    """Lanes per workgroup."""
    max_grid: tuple = ()
    """The largest grid its ranges are proven for (empty: no bound)."""
    host: object = None
    """``host(bind, **given) -> Bound``, the kernel's own binding, or None."""
    wg: int = 1
    """Workgroup ids: 1 (x) or 2 (x and y)."""
    preload: int = 0
    """Argument dwords in SGPRs at entry."""
    meta: dict = field(default_factory=dict)
    """What a library keeps with the definition."""
    grid: tuple = ()
    """The grid a bind without one launches (empty: bind needs a grid)."""

    def compile(self, build=None):
        """Builds and renders: derive, then compile_kernel; returns the Artifact.

        ``build`` is the build parameters' record for the manifest.
        """
        from thin import codegen, dsl

        def build_kernel():
            b = dsl.Builder()
            k = b.entry(self.args, wg=self.wg, preload=self.preload, wgsize=self.wgsize)
            self.body(b, k)
            return b

        return codegen.compile_kernel(
            codegen.derive(build_kernel),
            self.symbol,
            wgsize=self.wgsize,
            build=build,
            grid=tuple(self.grid),
            max_grid=tuple(self.max_grid),
        )

    def render(self, path) -> None:
        """Compiles, writes the text to path and prints the summary."""
        art = self.compile()
        with open(path, "w") as f:
            f.write(art.text)
        print(f"{path}: {art.summary}")

    def accepts(self, **given):
        """None when the host would bind these values, else why not."""
        from thin.runtime.bound import accepts

        return accepts(self, given)


@dataclass(frozen=True, eq=False)
class Kernel:
    """A rendered kernel."""

    name: str
    """The manifest's symbol."""
    asm: str
    """The .s text, its last line the manifest line."""
    manifest: object
    """The parsed manifest."""
    co: bytes
    """The code object."""
    params: dict
    """The bound build parameters."""
    address: str
    """The render's address: the kernel's identity."""
    definition: KernelDef
    """The KernelDef the generator returned."""

    @property
    def args(self) -> dict:
        """{name: argument record} from the manifest, in its order."""
        from thin.codegen import Manifest

        defaults = {"lo": None, "hi": None, "divisor": False, "multiple": 1, "owned": None}
        defaults.update(library=None, mem=None, ranks=None)
        recs = self.manifest.args if isinstance(self.manifest, Manifest) else self.manifest["args"]
        return {a["name"]: {**defaults, **a} for a in recs}

    def bind(self, grid=None, args=None, *, co_resident=False, overread=None):
        """A Bound: one call, not yet made."""
        from thin.runtime.bound import bind

        return bind(self, grid, args, co_resident=co_resident, overread=overread)

    def __call__(self, **given):
        """The kernel's own host."""
        host = self.definition.host
        if host is None:
            raise TypeError(
                f"{self.name}: no host function (KernelDef(host=)): bind it by hand, "
                "bind(grid=, args=)"
            )
        return host(self.bind, **given)

    def run(self, grid=None, args=None, **needs):
        """``bind(grid, args, **needs).run()``."""
        return self.bind(grid, args, **needs).run()


def cdiv(a, b):
    """``a / b`` rounded up (toward +inf), no zero check."""
    return -(-a // b)


_UNSET = object()
_STORE: list = [_UNSET]
"""The store's root, None, or not yet read (THIN_STORE is read when first needed)."""


def _default_store():
    """The default store.

    THIN_STORE when set and non-empty (``~`` expanded, ``off``: none), else
    ``~/.cache/thin-store``.
    """
    v = os.environ.get("THIN_STORE", "")
    if not v:
        return os.path.expanduser("~/.cache/thin-store")
    return None if v == "off" else os.path.expanduser(v)


def _store():
    if _STORE[0] is _UNSET:
        _STORE[0] = _default_store()
    return _STORE[0]


def use_store(root):
    """Sets the store compile reads and writes; returns the previous one."""
    before = _store()
    _STORE[0] = None if root is None else os.path.expanduser(root)
    return before


def _tupled(x):
    return tuple(_tupled(m) for m in x) if isinstance(x, (list, tuple)) else x


_ADDRESSES: dict = {}
"""Parts -> address, for parts that are hashable (a compile's are: tuples)."""


def render_address(identity, pairs, options, target, digest) -> str:
    """A render's address.

    ``address("render", FORMAT, identity, held pairs, options pairs, target, packages digest)``.
    """
    key = (identity, pairs, options, target, digest)
    try:
        found = _ADDRESSES.get(key)
    except TypeError:  # lists (as JSON test data passes them): not kept
        found, key = None, None
    if found is None:
        from thin.runtime.store import address

        parts = (_tupled(identity), _tupled(pairs), _tupled(options), target, digest)
        found = address("render", FORMAT, *parts)
        if key is not None:
            _ADDRESSES[key] = found
    return found


_SOURCES: dict = {}
"""Generator -> (packages digest, hashed files), once a process."""
_DIGESTS: dict = {}
"""The packages digest and hashes of a set of hashed tops, once a process."""


def _sources(generator) -> tuple[tuple, str, dict]:
    """(identity, packages digest, hashed files) of a generator.

    The file part and the digest once a process, the captures at every call.
    """
    from thin.runtime.identity import generator_identity

    identity = generator_identity(generator)
    found = _SOURCES.get(generator)
    if found is None:
        from thin.runtime import packages

        path = generator.__code__.co_filename
        hashed, versions = packages.packages_for([__file__, path])
        key = (tuple(sorted(hashed)), tuple(sorted(versions)))
        found = _DIGESTS.get(key)
        if found is None:
            memo: dict = {}
            digest = packages.packages_digest(hashed, versions, memo)
            found = (digest, packages.hashes(hashed, memo))
            _DIGESTS[key] = found
        _SOURCES[generator] = found
    return (identity, *found)


_KERNELS: dict[str, Kernel] = {}


def compile(generator, **params) -> Kernel:
    """The Kernel of a generator at these parameters, through the store."""
    from thin.target import target

    bound = bind_params(generator, params)
    pairs = held_pairs(bound)
    identity, digest, sources = _sources(generator)
    from thin._options import output_pairs

    addr = render_address(identity, pairs, output_pairs(), target().name, digest)
    found = _KERNELS.get(addr)
    if found is not None:
        return found
    d = declare(generator, **params)

    def render():
        from thin import codegen

        text = d.compile(build=dict(pairs)).text
        return {"s": text.encode(), "co": codegen.code_object(text)}

    root = _store()
    if root is None:
        members = render()
    else:
        from thin.runtime import store

        members = store.get(root, addr, render, ("s", "co"), wait=_WAIT, sources=sources)
    return _keep(generator, bound, addr, d, members)


def _keep(generator, bound, addr, d, members) -> Kernel:
    from thin.codegen import split_manifest

    asm = members["s"].decode()
    _, manifest = split_manifest(asm)
    if manifest is None:
        raise ValueError(
            f"{generator.__qualname__}: the .s has no manifest line (emit_kernel writes it)"
        )
    k = Kernel(manifest.symbol, asm, manifest, members["co"], bound, addr, d)
    _KERNELS[addr] = k
    return k


def forget() -> None:
    """Forgets this process's Kernels (a test's second process in this one)."""
    _KERNELS.clear()
