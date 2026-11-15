"""Golden checks of compilation, binding, execution and store interfaces."""

from __future__ import annotations

import ast
import contextlib
import functools
import hashlib
import struct
import subprocess
import sys

import pytest

from tests.golden import cases

PARAMS = ("vector", "case_id", "case", "doc")


def _raises(case, fn):
    """Checks a case's ``raises``/``message`` (or ``error``, a LaunchError) against fn()."""
    import builtins

    from thin.runtime import errors

    kind = case.get("raises", "LaunchError")
    want = case.get("message", case.get("error"))
    cls = getattr(errors, kind, None) or getattr(builtins, kind)
    with pytest.raises(cls) as e:
        fn()
    assert e.value.args[0] == want
    return e.value


# --- arith.json --------------------------------------------------------------


def _expr(v):
    if isinstance(v, dict):
        if "t" in v:
            return tuple(_expr(m) for m in v["t"])
        if "list" in v:
            return [_expr(m) for m in v["list"]]
        if "bool" in v:
            return v["bool"]
        if "float" in v:
            return float(v["float"])
    return v


@pytest.mark.parametrize(PARAMS, cases("runtime", "arith.json"))
def test_arith(vector, case_id, case, doc):
    from thin.runtime import arith

    section = case_id.split("[")[0]
    if section == "magic":
        assert arith.magic(case["d"]) == (case["mul"], case["shift"])
    elif section == "magic_refused":
        _raises(case, lambda: arith.magic(_expr(case["d"])))
    else:
        assert section == "cdiv", section
        if "error" in case:
            _raises(case, lambda: arith.cdiv(case["a"], case["b"]))
        else:
            assert arith.cdiv(case["a"], case["b"]) == case["value"]


# --- args.json ---------------------------------------------------------------


class _Buf:
    """A value the check admits as a buffer."""

    def __repr__(self) -> str:
        return "<buffer>"


def _arg(v):
    import numpy as np

    if isinstance(v, dict):
        if "buffer" in v:
            return _Buf()
        if "bytes" in v:
            return bytes.fromhex(v["bytes"])
        if "bool" in v:
            return v["bool"]
        if "float" in v:
            return float(v["float"])
        if "numpy" in v:
            return getattr(np, v["numpy"])(v["value"])
    return v


@pytest.mark.parametrize(PARAMS, cases("runtime", "args.json"))
def test_args(vector, case_id, case, doc):
    from thin.runtime import args

    section = case_id.split("[")[0]
    if section == "blocks":
        assert case[-1]["offset"] + case[-1]["bytes"] <= doc["sizes"][case_id[7:-1]]
    elif section == "sizes":
        block = doc["blocks"][case_id[6:-1]]
        assert max(r["offset"] + r["bytes"] for r in block) <= case
    elif section == "check":
        values = {n: _arg(v) for n, v in case["values"].items()}
        records = doc["blocks"][case["block"]]

        def run():
            args.check("k", records, values, lambda v: isinstance(v, _Buf))

        if case["error"] is None:
            run()
        else:
            _raises(case, run)
    else:
        values = {n: _arg(v) for n, v in case["values"].items()}
        records = doc["blocks"][case["block"]]

        def run():
            return args.pack(records, case["nbytes"], values, lambda n: case["addresses"][n])

        if "hex" in case:
            assert run().hex() == case["hex"]
        else:
            _raises(case, run)


# --- store.json ------------------------------------------------------------


def _dv(v):
    if "s" in v:
        return v["s"]
    if "b" in v:
        return bytes.fromhex(v["b"])
    if "i" in v:
        return v["i"]
    return tuple(_dv(m) for m in v["t"])


@pytest.mark.parametrize(PARAMS, cases("runtime", "store.json"))
def test_store(vector, case_id, case, doc, tmp_path):
    from thin.runtime import packages, store

    section = case_id.split("[")[0]
    if section == "digests":
        values = [_dv(v) for v in case["values"]]
        assert store.encode(*values).hex() == case["encoding"]
        assert store.digest(*values) == case["digest"]
    elif section == "digest_refused":
        _raises(
            {**case, "raises": "TypeError"}, lambda: store.digest(ast.literal_eval(case["value"]))
        )
    elif section == "addresses":
        assert store.address(case["name"], *map(_dv, case["values"])) == case["address"]
    elif section == "bad_names":
        _raises({**case, "raises": "ValueError"}, lambda: store.address(case["name"], "x"))
    elif section == "not_addresses":
        _raises({**case, "raises": "ValueError"}, lambda: store.path_of(tmp_path, case["address"]))
    elif section == "entry":
        _entry(case_id[6:-1], case, doc["entry"], tmp_path)
    elif section == "pack_refused":
        members = ast.literal_eval(case["members"])
        addr = doc["entry"]["address"]
        _raises({**case, "raises": "TypeError"}, lambda: store.entry(addr, members))
    elif section == "lookups":
        path = store.path_of(tmp_path, case["address"])
        __import__("os").makedirs(__import__("os").path.dirname(path))
        with open(path, "wb") as f:
            f.write(bytes.fromhex(case["file_hex"]))
        got = store.lookup(tmp_path, case["address"], case["names"])
        if isinstance(case["result"], str):
            assert got is store.Miss(case["result"])
        else:
            assert {n: v.hex() for n, v in got.items()} == case["result"]
    elif section == "packages_digest":
        p = doc["packages_digest"]
        files = tuple(tuple(f) for f in p["files"])
        want = store.digest(
            (files, tuple(tuple(v) for v in p["versions"]), tuple(p["interpreter"]))
        )
        assert want == p["digest"]
        assert packages.interpreter()[0] == "cpython"
    elif case_id in ("magic", "head_bytes"):
        pass
    else:
        raise AssertionError(case_id)


def test_store_magic_and_head():
    from tests.golden import load
    from thin.runtime import store

    doc = load("runtime", "store.json")
    assert doc["magic"].encode() == store.MAGIC
    assert doc["head_bytes"] == store.HEAD


def test_store_header_refused():
    from tests.golden import load
    from thin.runtime import store

    doc = load("runtime", "store.json")
    addr = doc["header_refused"].split(":")[0]
    with pytest.raises(ValueError) as e:
        store.entry(addr, {f"m{i}": b"" for i in range(1000)})
    assert str(e.value) == doc["header_refused"]


def _entry(key, value, e, tmp_path):
    from thin.runtime import store

    members = {n: bytes.fromhex(v) for n, v in e["members"].items()}
    if key == "file_hex":
        assert store.entry(e["address"], members).hex() == value
    elif key == "content_digest":
        assert store.content_digest(members) == value
    elif key == "path":
        assert store.path_of("<root>", e["address"]) == value
    elif key == "header":
        blob = store.entry(e["address"], members)
        head = blob[len(store.MAGIC) :].split(b"\n", 1)[0]
        import json

        assert json.loads(head) == value


# --- kernel.json ----------------------------------------------------------------


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


def _given(params: dict) -> dict:
    import numpy as np

    from thin.opspec import Field

    scope = {"np": np, "Field": Field, "frozenset": frozenset, "set": set, "object": object}
    out = {}
    for n, text in params.items():
        out[n] = object if text == "<class 'object'>" else eval(text, scope)
    return out


def _pairs(given):
    from thin.runtime.params import bind_params, held_pairs

    return [list(p) for p in held_pairs(bind_params(gen_toy, given))]


@functools.cache
def _vadd_store(root: str):
    """vadd compiled through a fresh store; the report's counts for it."""
    import thin
    from examples.vadd import build_vadd
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
    counts = {
        k: v for k, v in store.report().items() if k.startswith(first.address.split("/")[0] + ".")
    }
    return first, same, again, counts, root


@pytest.fixture(scope="module")
def vadd(tmp_path_factory):
    return _vadd_store(str(tmp_path_factory.mktemp("store")))


def _arrays():
    import numpy as np

    return {n: np.zeros(4, np.float32) for n in "xyz"}


def _needy(same_grid=False):
    from thin import KernelDef
    from thin.dsl import Args, kept, ptr, u32

    def body(b, k):
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


def _closure_case(c, folder):
    """One `closures` case: src written to closure_gen.py, imported, `call` run."""
    import importlib
    import sys

    from thin.runtime import generator_identity

    folder.mkdir(parents=True)
    (folder / "closure_gen.py").write_bytes(c["src"].encode())
    for name, text in c.get("files", {}).items():
        (folder / name).write_bytes(text.encode())
    sys.path.insert(0, str(folder))
    for gone in ("closure_gen", *(n[:-3] for n in c.get("files", {}))):
        sys.modules.pop(gone, None)
    try:
        module = importlib.import_module("closure_gen")
        scope = {**vars(module), "generator_identity": generator_identity}
        if "raises" in c:
            with pytest.raises(TypeError) as e:
                exec(c["call"], scope)
            assert str(e.value) == c["message"], c["what"]
        else:
            exec(c["call"], scope)
            r = scope["r"]
            assert [r[0], r[1], r[2], [list(p) for p in r[3]]] == c["ok"], c["what"]
    finally:
        sys.path.remove(str(folder))
        for gone in ("closure_gen", *(n[:-3] for n in c.get("files", {}))):
            sys.modules.pop(gone, None)


@pytest.mark.parametrize(PARAMS, cases("runtime", "kernel.json"))
def test_kernel(vector, case_id, case, doc, vadd, tmp_path):
    import numpy as np

    import thin
    from thin.runtime import kernel
    from thin.runtime.errors import LaunchError

    section = case_id.split("[")[0]
    first, same, again, counts, root = vadd
    if section == "params":
        assert _pairs(_given(case["given"])) == case["ok"]
    elif section == "refusals":
        given = _given(case["given"])
        _raises(case, lambda: _pairs(given))
    elif section == "overriding":
        key = case_id[11:-1]
        with thin.overriding({gen_toy: {"BLOCK": 64, "scale": 3.0}}):
            outer = _pairs({"BLOCK": 128})
            with thin.overriding({gen_toy: {"BLOCK": 32}}):
                inner = _pairs({"BLOCK": 128})
        after = _pairs({"BLOCK": 128})
        assert {"outer": outer, "inner": inner, "after": after}[key] == case
    elif section == "render_address":
        from thin._options import output_pairs

        for r in doc["render_address"]["cases"]:
            assert r["format"] == kernel.FORMAT, r["what"]
            with thin.codegen.using(**r["using"]):
                assert [list(p) for p in output_pairs()] == r["options"], r["what"]
            made = kernel.render_address(
                r["identity"], r["held"], r["options"], r["target"], r["packages_digest"]
            )
            assert made == r["address"], r["what"]
    elif section == "closures":
        for c in doc["closures"]["cases"]:
            _closure_case(c, tmp_path / str(abs(hash(c["what"]))))
    elif section == "store_round_trip":
        key = case_id[17:-1]
        if key == "same_object_in_process":
            assert (first is same) == case
        elif key == "counts":
            assert counts == case
        elif key == "equal_after_forgetting":
            assert (again.asm, again.co, again.address) == (first.asm, first.co, first.address)
            assert again is not first
        else:
            from thin.runtime import store

            assert sorted(store.lookup(root, first.address)) == case
    elif section == "vadd":
        _vadd_fact(case_id[5:-1], case, first)
    elif section == "bind":
        _bind_case(case, first)
    elif section == "host":
        a = _arrays()
        n = 1000 if "ok" in case else 0
        if "ok" in case:
            assert list(first(**a, n=n).grid) == case["ok"]
        else:
            _raises(case, lambda: first(**a, n=n))
    elif section == "accepts":
        d = first.definition
        given = {
            k: (np.zeros(4, np.float32) if v == "<array>" else v) for k, v in case["given"].items()
        }
        if "raises" in case:
            _raises(case, lambda: d.accepts(**given))
        else:
            assert d.accepts(**given) == case["ok"]
    elif section == "needs":
        k = thin.compile(gen_needy)
        key = case_id[6:-1]
        if key == "needs":
            assert k.manifest.needs == case
        elif key == "manifest_line":
            assert k.asm.rstrip("\n").splitlines()[-1] == case.rstrip("\n")
        elif key == "bind":
            for c in case:
                args = {"x": np.zeros(4, np.float32), "c": 16, "n": 2}
                if "ok" in c:
                    assert list(k.bind(2, args, co_resident=True).grid) == c["ok"]
                else:
                    _raises(c, lambda args=args: k.bind(2, args))
        else:
            _raises(case, lambda: k())
    elif section == "same_grid":
        k = thin.compile(gen_needy_same_grid)
        assert k.manifest.needs["same_grid"] == case["same_grid"]
    elif section == "kernarg":
        key = case_id[8:-1]
        if key == "hex":
            b = first.bind(4, {**_arrays(), "n": 1000})
            assert b.kernarg(lambda n: doc["kernarg"]["addresses"][n]).hex() == case
    else:
        raise AssertionError(case_id)
    assert LaunchError


def _vadd_fact(key, value, k):
    if key == "params":
        assert {n: repr(v) for n, v in k.params.items()} == value
    elif key == "name":
        assert k.name == value
    elif key == "address_name":
        assert k.address.split("/")[0] == value
    elif key == "address_hex_len":
        assert len(k.address.split("/")[1]) == value
    elif key == "args":
        got = {n: [r["offset"], r["bytes"], r["kind"], r["lo"], r["hi"]] for n, r in k.args.items()}
        assert got == value
    elif key == "kernarg_bytes":
        assert k.manifest.kernarg_bytes == value
    elif key in ("grid", "max_grid", "build"):
        assert getattr(k.manifest, key) == value
    elif key == "s_sha256":
        assert hashlib.sha256(k.asm.encode()).hexdigest() == value
    elif key == "co_sha256":
        assert hashlib.sha256(k.co).hexdigest() == value
    elif key == "manifest_line":
        assert k.asm.rstrip("\n").splitlines()[-1] == value.rstrip("\n")
    else:
        raise AssertionError(key)


def _bind_case(case, k):
    import numpy as np

    call = case["call"]
    all_args = {**_arrays(), "n": 1000}
    args = {n: all_args[n] for n in call["args"]}
    kw = {}
    if "overread" in call:
        kw["overread"] = ast.literal_eval(call["overread"])
    if "co_resident" in call:
        kw["co_resident"] = ast.literal_eval(call["co_resident"])
    grid = eval(call["grid"], {"np": np}) if "grid" in call else None

    def run():
        return k.bind(grid, args, **kw)

    if "ok" in case:
        b = run()
        assert {"grid": list(b.grid), "co_resident": b.co_resident, "overread": b.overread} == case[
            "ok"
        ]
    else:
        _raises(case, run)


# --- device.json ---------------------------------------------------------------


def _namespace():
    """The names device.json's notation binds."""
    import thin.runtime as runtime
    from thin import formats

    def f32(*xs):
        return struct.pack(f"<{len(xs)}f", *xs)

    d = formats.DTYPES
    return {"runtime": runtime, "F32": d["f32"], "BF16": d["bf16"], "U8": d["u8"], "f32": f32}


def _run_section(doc, section, index):
    """Runs the section's setup and the calls before ``index`` (as the replay
    runs them in order), then case ``index``'s call; returns the namespace."""
    scope = _namespace()
    exec(doc[f"{section}_setup"], scope)
    for before in doc[section][:index]:
        with contextlib.suppress(Exception):  # an earlier case's refusal is that case's
            exec(before["call"], scope)
    scope.pop("r", None)
    return scope


def _device_case(case_id, case, doc):
    from thin.runtime.errors import DeviceFault

    section, index = case_id.split("[")[0], int(case_id.split("[")[1][:-1])
    scope = _run_section(doc, section, index)
    if "raises" in case:
        e = _raises(case, lambda: exec(case["call"], scope))
        if "cause" in case:
            assert isinstance(e, DeviceFault) and e.cause == case["cause"]
    else:
        exec(case["call"], scope)
        r = scope.get("r")
        if isinstance(r, bytes):
            r = r.hex()
        elif isinstance(r, list):
            r = None  # a group that was made
        assert r == case["ok"]
    if "synced" in case:
        assert scope["h"].synced[-1] == case["synced"]


@pytest.mark.parametrize(PARAMS, cases("runtime", "device.json", "steps"))
def test_device_steps(vector, case_id, case, doc):
    _device_case(case_id, case, doc)


@pytest.mark.parametrize(PARAMS, cases("runtime", "device.json", "guards"))
def test_device_guards(vector, case_id, case, doc):
    _device_case(case_id, case, doc)


# --- Import cost ----------------------------------------------------------------------


def _modules_after(statement: str) -> set[str]:
    code = f"import sys; {statement}; print('\\n'.join(sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    return set(out.stdout.split())


_COMPILER = ("thin.dsl", "thin.codegen", "thin.ir", "thin.target", "thin.emu", "thin._native")


def test_root_import_loads_nothing_behind_its_names():
    loaded = _modules_after("import thin")
    assert not {m for m in loaded if m.startswith("thin.")}


def test_runtime_definitions_load_no_compiler_nor_numpy():
    stmt = "import thin.runtime, thin.runtime.kernel, thin.runtime.params, thin.runtime.bound"
    loaded = _modules_after(stmt)
    assert not {m for m in loaded if m.startswith(_COMPILER) or m in {"numpy", "torch"}}


def test_hip_module_loads_only_itself_and_the_errors():
    loaded = _modules_after("import thin.runtime.hip")
    thin = {m for m in loaded if m.startswith("thin")}
    assert thin == {"thin", "thin.runtime", "thin.runtime.errors", "thin.runtime.hip"}
    assert not loaded & {"numpy", "torch"}


def test_root_exposes_the_opspec_names():
    """with list, candidates and lowered included."""
    import thin

    names = {"Spec", "Int", "Choice", "Bool", "Field", "Tensor", "In", "Out", "InOut"}
    names |= {"implements", "implementations", "candidates", "lower", "lowered", "check", "group"}
    assert names <= set(thin.__all__)


def test_root_unknown_attribute():
    import thin

    with pytest.raises(AttributeError, match="module 'thin' has no attribute 'nope'"):
        thin.nope  # noqa: B018
