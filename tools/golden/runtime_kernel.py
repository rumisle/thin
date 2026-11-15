"""kernel.json: build parameters, render addresses, closures, compile/bind/host.

The authored inputs here (parameter texts, address parts, bind calls, accepts
arguments, kernarg addresses) are new; every recorded value comes from thin:
held pairs, addresses, compiled vadd and needy kernels, Bound grids and exact
refusals. The kernels compile through a fresh temporary store.

``gen_toy``, ``gen_needy`` and ``gen_needy_same_grid`` mirror the replay's
source helpers of the same names (signature and body), since their names,
defaults and rendered text are part of the recorded results.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import tempfile
from pathlib import Path

from tools.golden import ROOT, runtime_closures


def gen_toy(
    BLOCK: int = 256,  # noqa: N803 - the vectors' name
    n: tuple = (1, 100),
    ablate=frozenset(),
    scale: float = 1.0,
    tag=None,
    name="a",
):
    """The params vectors' generator: a KernelDef that is never rendered."""
    from thin import KernelDef
    from thin.dsl import Args, ptr

    return KernelDef(f"toy_{name}", Args(x=ptr), lambda b, k: None, wgsize=BLOCK)


def _needy(same_grid=False):
    from thin import KernelDef
    from thin.dsl import Args, kept, ptr, u32

    def body(b, k):
        """Construct body inputs."""
        b.comment("nothing")  # the stream's one item: its peak line
        b.need("co_resident", "its workgroups wait for each other")
        if same_grid:
            b.need("same_grid", "its count words are per workgroup", "c")

    return KernelDef(
        "needy", Args(x=ptr, c=kept, n=u32.within(1, 64)), body, wgsize=32, max_grid=(8, 2)
    )


def gen_needy():
    """The needs vectors' kernel: it asks co_resident."""
    return _needy()


def gen_needy_same_grid():
    """gen_needy, also asking the same grid on c."""
    return _needy(True)


NOTATION = (
    "Compile, Kernel and bind on {target}. `params`: the generator `gen_toy(BLOCK: int = 256, "
    "n: tuple = (1, 100), ablate=frozenset(), scale: float = 1.0, tag=None, name='a')` given "
    "parameters, each written as Python text (evaluated with np, Field, frozenset, set and "
    "object in scope; the text \"<class 'object'>\" is the class object), and `ok` its held "
    "pairs [name, repr of the held form], sorted by name; `refusals` are given parameters that "
    "are refused. `overriding`: gen_toy given BLOCK=128 inside overriding(BLOCK=64, scale=3.0), "
    "inside that and a nested overriding(BLOCK=32), and after both. `render_address`: addresses "
    "made from stand-in parts. `closures`: generator identities of written modules. `vadd`: "
    "examples.vadd's build_vadd at its defaults (z = x + y over n f32, BLOCK lanes a workgroup), "
    "compiled through a fresh store, and `store_round_trip` that store's behavior. `bind`: "
    "Kernel.bind calls on vadd (`grid` text evaluated with np in scope; `args` the names given, "
    "x, y, z as 4-element f32 arrays and n = 1000; `overread`/`co_resident` as Python text) and "
    "the Bound's grid, co_resident and overread or the refusal. `host`: vadd's own host given "
    "the three arrays and n = 1000 (`ok`, the grid) or n = 0 (refused). `accepts`: "
    'KernelDef.accepts on the given values ("<array>" a 4-element f32 array). `needs`: the '
    "`needy` kernel that asks co_resident (bind args x, c = 16, n = 2 at grid 2, with "
    "co_resident=True when `ok`), and `same_grid` the variant also asking the same grid on c. "
    "`kernarg`: the vadd Bound of grid 4 (n = 1000) packed with these pointer addresses. `ok` "
    "holds a result; `raises`/`message` an exception."
)

PARAMS = [
    {},
    {"name": "'a'"},
    {"BLOCK": "1024", "n": "[2, 48]"},
    {"ablate": "{'stores', 'lds', 'atomics'}"},
    {"ablate": "frozenset({'waits', 'barriers'})"},
    {"ablate": "('waits', 'barriers')"},
    {"ablate": "frozenset()"},
    {"tag": "()"},
    {"ablate": "{9, 10}"},
    {"scale": "7"},
    {"scale": "0.25"},
    {"scale": "-0.0"},
    {"tag": "None", "scale": "1.0"},
    {"tag": "False"},
    {"tag": "0"},
    {"name": '"it\'s"'},
    {"name": "'µs'"},
    {"n": "(4, [5, frozenset({7, 6})])"},
]

REFUSALS = [
    {"scale": "Field(12, 0, 31)"},
    {"ablate": "{'k': 2}"},
    {"name": "bytes(2)"},
    {"n": "(1, np.float32(0.5))"},
    {"WARPS": "2", "BLOCK": "64"},
    {"tag": "int"},
    {"tag": "<class 'object'>"},
    {"tag": "len"},
    {"tag": "object()"},
]

# Stand-in identity and digest: digests of labels, not of real files.
_SHA = hashlib.sha256(b"stand-in generator source").hexdigest()
_PKG = hashlib.sha256(b"stand-in packages").hexdigest()
_BASE = {
    "identity": ["make_saxpy", _SHA, 17, []],
    "held": [["BLOCK", "128"], ["unroll", "2"]],
}

RENDERS = [
    ("the default options", {}, {}, None),
    *[
        (
            f"using({o}={v}): another address",
            {o: v},
            {},
            ("differs", "all"),
        )
        for o, v in (
            ("audit", True),
            ("keep_dead", True),
            ("safe_barrier", True),
            ("waits_unique", True),
            ("safe_waits", True),
            ("regions", False),
            ("lds_wave_order", False),
        )
    ],
    (
        "using(peak_report=True): the defaults' address (it changes no output)",
        {"peak_report": True},
        {},
        ("same", 0),
    ),
    (
        "using(pressure=True, peak_list=3): the defaults' address",
        {"pressure": True, "peak_list": 3},
        {},
        ("same", 0),
    ),
    (
        "using(pressure=True, safe_waits=True): safe_waits' address",
        {"pressure": True, "safe_waits": True},
        {},
        ("same", 5),
    ),
    (
        "a closure capturing lanes = 32",
        {},
        {"identity": ["make_saxpy", _SHA, 17, [["lanes", "32"]]]},
        ("differs", "all"),
    ),
    (
        "the same factory's closure capturing lanes = 96: another address",
        {},
        {"identity": ["make_saxpy", _SHA, 17, [["lanes", "96"]]]},
        ("differs", "all"),
    ),
    (
        "two captures in name order, a list held as a tuple",
        {},
        {"identity": ["make_saxpy", _SHA, 17, [["lanes", "32"], ["tile", "(4, 8)"]]]},
        ("differs", "all"),
    ),
    (
        "the same qualified name defined at line 53: another address",
        {},
        {"identity": ["make_saxpy", _SHA, 53, []]},
        ("differs", "all"),
    ),
    (
        "another held parameter value: another address",
        {},
        {"held": [["BLOCK", "128"], ["unroll", "4"]]},
        ("differs", "all"),
    ),
    (
        "another target name: another address",
        {},
        {"target": "gfx1100"},
        ("differs", "all"),
    ),
    (
        "another packages digest: another address",
        {},
        {"packages_digest": hashlib.sha256(b"other packages").hexdigest()},
        ("differs", "all"),
    ),
]

_ALL = ["n", "x", "y", "z"]
_MAX = "-(-((1 << 30) - 1) // 256)"
"""vadd's max_grid as text: its n range's top over 256 lanes, rounded up."""

BINDS = [
    ("a grid of 9", {"grid": "9", "args": _ALL}, "ok"),
    ("a 3-d grid", {"grid": "(6, 1, 1)", "args": _ALL}, "ok"),
    ("a list grid", {"grid": "[3, 1]", "args": _ALL}, "ok"),
    ("a numpy uint32 grid", {"grid": "np.uint32(5)", "args": _ALL}, "ok"),
    ("a numpy int16 in a grid", {"grid": "(np.int16(2), 1)", "args": _ALL}, "ok"),
    ("a float in y of the grid", {"grid": "(3, 1.0)", "args": _ALL}, "refusal"),
    ("the grid at max_grid", {"grid": _MAX, "args": _ALL}, "ok"),
    (
        "one past max_grid",
        {"grid": f"({_MAX} + 1, 1)", "args": _ALL},
        "refusal",
    ),
    ("z past max_grid's absent z (1)", {"grid": "(1, 1, 2)", "args": _ALL}, "refusal"),
    ("a zero y", {"grid": "(7, 0)", "args": _ALL}, "refusal"),
    ("a negative grid", {"grid": "-2", "args": _ALL}, "refusal"),
    ("four grid dimensions", {"grid": "(2, 1, 1, 1)", "args": _ALL}, "refusal"),
    ("no grid dimensions", {"grid": "()", "args": _ALL}, "refusal"),
    ("a bool grid", {"grid": "True", "args": _ALL}, "refusal"),
    ("no grid and none in the manifest", {"args": _ALL}, "refusal"),
    ("two pointers missing", {"grid": "2", "args": ["n", "y"]}, "refusal"),
    (
        "an overread of z",
        {"grid": "3", "args": _ALL, "overread": "{'z': 64}"},
        "ok",
    ),
    (
        "overreads of two pointers",
        {"grid": "3", "args": _ALL, "overread": "{'y': 4, 'z': 12}"},
        "ok",
    ),
    (
        "an overread of n",
        {"grid": "3", "args": _ALL, "overread": "{'n': 8}"},
        "refusal",
    ),
    (
        "an overread of 0 bytes of y",
        {"grid": "3", "args": _ALL, "overread": "{'y': 0}"},
        "refusal",
    ),
    (
        "an overread of True bytes",
        {"grid": "3", "args": _ALL, "overread": "{'x': True}"},
        "refusal",
    ),
    (
        "an overread of a name vadd lacks",
        {"grid": "3", "args": _ALL, "overread": "{'out': 4}"},
        "refusal",
    ),
    (
        "co_resident given as 1, no need asked",
        {"grid": "5", "args": _ALL, "co_resident": "1"},
        "ok",
    ),
    (
        "co_resident=False",
        {"grid": "5", "args": _ALL, "co_resident": "False"},
        "ok",
    ),
    ("a scalar float grid", {"grid": "4.0", "args": _ALL}, "refusal"),
]

ACCEPTS = [
    ({"n": 1}, "accepted"),
    ({"n": (1 << 30) - 1}, "accepted"),
    ({"n": 0}, "refused"),
    ({"n": -7}, "refused"),
    ({"n": 1 << 30}, "refused"),
    ({"x": "<array>", "y": "<array>", "z": "<array>", "n": 77}, "accepted"),
    ({"x": "<array>", "n": 9}, "accepted"),
    ({"x": "<array>", "y": 7.5, "z": "<array>", "n": 9}, "refused"),
    ({"z": 12, "n": 9}, "refused"),
    ({}, "raises"),
    ({"n": 3, "w": 1}, "raises"),
]

KERNARG = {"x": 0x7F3A_0010_0000, "y": 0x7F3A_0010_4000, "z": 0x7F3A_0020_0040}


def _given(params: dict) -> dict:
    import numpy as np

    from thin.opspec import Field

    scope = {"np": np, "Field": Field, "frozenset": frozenset, "set": set, "object": object}
    return {
        n: object if text == "<class 'object'>" else eval(text, scope) for n, text in params.items()
    }


def _pairs(given):
    from thin.runtime.params import bind_params, held_pairs

    return [list(p) for p in held_pairs(bind_params(gen_toy, given))]


def _refusal(e) -> dict:
    import builtins

    from thin.runtime import errors

    name = type(e).__name__
    if (getattr(errors, name, None) or getattr(builtins, name, None)) is not type(e):
        raise AssertionError(f"{name} is not nameable by the replay") from e
    return {"raises": name, "message": e.args[0]}


def _attempt(fn) -> dict:
    try:
        return {"ok": fn()}
    except Exception as e:  # refusals are recorded, never written
        return _refusal(e)


def _arrays():
    import numpy as np

    return {n: np.zeros(4, np.float32) for n in "xyz"}


def _params():
    params = [{"given": g, "ok": _pairs(_given(g))} for g in PARAMS]
    refusals = []
    for g in REFUSALS:
        got = _attempt(lambda g=g: _pairs(_given(g)))
        if "ok" in got:
            raise AssertionError(f"refusal {g} was accepted")
        refusals.append({"given": g, **got})
    return params, refusals


def _overriding():
    import thin

    with thin.overriding({gen_toy: {"BLOCK": 64, "scale": 3.0}}):
        outer = _pairs({"BLOCK": 128})
        with thin.overriding({gen_toy: {"BLOCK": 32}}):
            inner = _pairs({"BLOCK": 128})
    return {"outer": outer, "inner": inner, "after": _pairs({"BLOCK": 128})}


def _renders():
    from thin._options import output_pairs, using
    from thin.runtime import kernel
    from thin.target import target

    cases = []
    for what, use, changes, relation in RENDERS:
        parts = {**_BASE, "target": target().name, "packages_digest": _PKG, **changes}
        with using(**use):
            options = [list(p) for p in output_pairs()]
        address = kernel.render_address(
            parts["identity"], parts["held"], options, parts["target"], parts["packages_digest"]
        )
        if relation is not None:
            kind, at = relation
            if kind == "same" and address != cases[at]["address"]:
                raise AssertionError(f"{what}: not {cases[at]['what']}'s address")
            if kind == "differs" and address in {c["address"] for c in cases}:
                raise AssertionError(f"{what}: an earlier case's address")
        cases.append(
            {
                "what": what,
                "using": use,
                "format": kernel.FORMAT,
                "identity": parts["identity"],
                "held": parts["held"],
                "options": options,
                "target": parts["target"],
                "packages_digest": parts["packages_digest"],
                "address": address,
            }
        )
    return cases


def _vadd(root: str, build_vadd):
    """Vadd through a fresh store, as the replay compiles it; the store's counts."""
    import thin
    from thin.runtime import kernel, store

    before = thin.use_store(root)
    kernel.forget()
    try:
        first = thin.compile(build_vadd)
        same = thin.compile(build_vadd, BLOCK=256)
        kernel.forget()
        again = thin.compile(build_vadd)
    finally:
        thin.use_store(before)
    prefix = first.address.split("/")[0] + "."
    counts = {k: v for k, v in store.report().items() if k.startswith(prefix)}
    if (again.asm, again.co, again.address) != (first.asm, first.co, first.address):
        raise AssertionError("vadd differs after forgetting")
    trip = {
        "same_object_in_process": first is same,
        "counts": counts,
        "equal_after_forgetting": again is not first,
        "entry_members": sorted(store.lookup(root, first.address)),
    }
    k = first
    facts = {
        "params": {n: repr(v) for n, v in k.params.items()},
        "name": k.name,
        "address_name": k.address.split("/")[0],
        "address_hex_len": len(k.address.split("/")[1]),
        "manifest_line": k.asm.rstrip("\n").splitlines()[-1],
        "args": {
            n: [r["offset"], r["bytes"], r["kind"], r["lo"], r["hi"]] for n, r in k.args.items()
        },
        "kernarg_bytes": k.manifest.kernarg_bytes,
        "grid": k.manifest.grid,
        "max_grid": k.manifest.max_grid,
        "build": k.manifest.build,
        "s_sha256": hashlib.sha256(k.asm.encode()).hexdigest(),
        "co_sha256": hashlib.sha256(k.co).hexdigest(),
    }
    return first, trip, facts


def _bind(k):
    import numpy as np

    cases = []
    for what, call, kind in BINDS:
        all_args = {**_arrays(), "n": 1000}
        args = {n: all_args[n] for n in call["args"]}
        kw = {}
        for key in ("overread", "co_resident"):
            if key in call:
                kw[key] = ast.literal_eval(call[key])
        grid = eval(call["grid"], {"np": np}) if "grid" in call else None

        def run(grid=grid, args=args, kw=kw):
            """Construct run inputs."""
            b = k.bind(grid, args, **kw)
            return {"grid": list(b.grid), "co_resident": b.co_resident, "overread": b.overread}

        got = _attempt(run)
        if ("ok" in got) != (kind == "ok"):
            raise AssertionError(f"{what}: planned {kind}, got {got}")
        cases.append({"what": what, "call": call, **got})
    return cases


def _host(k):
    ok = {"what": "k(**given) runs the host", "ok": list(k(**_arrays(), n=1000).grid)}
    refused = _attempt(lambda: k(**_arrays(), n=0))
    if "ok" in refused:
        raise AssertionError("vadd's host bound n = 0")
    return [ok, {"what": "the host's bind refuses", **refused}]


def _accepts(k):
    import numpy as np

    cases = []
    d = k.definition
    for given, kind in ACCEPTS:
        values = {n: (np.zeros(4, np.float32) if v == "<array>" else v) for n, v in given.items()}
        got = _attempt(lambda values=values: d.accepts(**values))
        found = "raises" if "raises" in got else "accepted" if got["ok"] is None else "refused"
        if found != kind:
            raise AssertionError(f"accepts {given}: planned {kind}, got {got}")
        cases.append({"given": given, **got})
    return cases


def _needs(needy_generator, same_grid_generator):
    import numpy as np

    import thin

    k = thin.compile(needy_generator)
    binds = []
    for co_resident in (False, True):
        args = {"x": np.zeros(4, np.float32), "c": 16, "n": 2}
        if co_resident:
            got = {"ok": list(k.bind(2, args, co_resident=True).grid)}
            binds.append({"what": "needs co_resident, given", **got})
        else:
            got = _attempt(lambda args=args: k.bind(2, args))
            if "ok" in got:
                raise AssertionError("needy bound without co_resident")
            binds.append({"what": "needs co_resident, not given", **got})
    no_host = _attempt(lambda: k())
    if "ok" in no_host:
        raise AssertionError("needy ran a host")
    needs = {
        "needs": k.manifest.needs,
        "manifest_line": k.asm.rstrip("\n").splitlines()[-1],
        "bind": binds,
        "no_host": no_host,
    }
    same = thin.compile(same_grid_generator)
    return needs, {"needs": {"same_grid": same.manifest.needs["same_grid"]}}


def _source_generators(folder):
    """Compile source-only copies in an isolated loose-generator directory.

    Runtime addresses hash all importable packages beside a loose generator.
    The needy generators live in tools/golden, beside tests (and tests/data)
    at the repo root, so compiling them in place would hash tests/data too.
    Copy source (never fixtures), retaining the public example's implementation
    and the replay's fixed input constructors, so that package hashing cannot
    depend on any golden output. No package-hashing API is stubbed or bypassed.
    """
    folder.mkdir()
    source = (ROOT / "examples/vadd.py").read_text()
    source += "\n\n" + "\n\n".join(
        inspect.getsource(function) for function in (_needy, gen_needy, gen_needy_same_grid)
    )
    path = folder / "kernel_inputs.py"
    path.write_text(source)
    scope = {"__name__": "golden_runtime_kernels", "__file__": str(path)}
    exec(compile(source, str(path), "exec"), scope)
    return scope


def generate_doc() -> dict:
    """Construct this document and record actual API outcomes."""
    import thin
    from thin.target import target

    params, refusals = _params()
    with tempfile.TemporaryDirectory(prefix="thin-golden-runtime-") as temp:
        root = Path(temp)
        generators = _source_generators(root / "input_sources")
        first, trip, facts = _vadd(str(root / "store"), generators["build_vadd"])
        before = thin.use_store(str(root / "needs-store"))
        try:
            needs, same_grid = _needs(generators["gen_needy"], generators["gen_needy_same_grid"])
        finally:
            thin.use_store(before)
        closures = runtime_closures.generate_section(root / "closures")
        b = first.bind(4, {**_arrays(), "n": 1000})
        kernarg = {"addresses": dict(KERNARG), "hex": b.kernarg(lambda n: KERNARG[n]).hex()}
        doc = {
            "notation": NOTATION.format(target=target().name),
            "params": params,
            "refusals": refusals,
            "overriding": _overriding(),
            "render_address": {
                "notation": (
                    'address("render", format, identity, held, options, target, '
                    "packages_digest) of the parts as given (tuples written as lists). "
                    "`options` are the output options' (name, repr) pairs of the Options in "
                    "force under `using`, sorted by name; the identity's sha and the packages "
                    "digest are stand-ins (sha256 of labels)."
                ),
                "cases": _renders(),
            },
            "store_round_trip": trip,
            "vadd": facts,
            "bind": _bind(first),
            "host": _host(first),
            "accepts": _accepts(first),
            "needs": needs,
            "same_grid": same_grid,
            "kernarg": kernarg,
            "closures": closures,
        }
    return doc
