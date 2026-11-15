"""Readers and writers for the Python tests' golden-data notation.

Each golden file describes its encoding in its ``notation`` field.
"""

from __future__ import annotations


def run_snippet(src: str, library: bool = False):
    """A fresh gfx1201 Builder after running a DSL snippet;
    with `library`, the library's modules in scope too."""
    from thin import dsl
    from thin.target import targeting

    with targeting("gfx1201"):
        b = dsl.Builder()
        scope = {name: getattr(dsl, name) for name in dsl.__all__}
        if library:
            from thin import lib

            scope.update(xwg=lib.xwg, comm=lib.comm, Laps=lib.Laps)
            scope.update(bf16_sum=lib.bf16_sum)
        scope.update(b=b, V=b.v, S=b.s)
        exec(src, scope)
    return b


def _operand(op, mod: int) -> str:
    from thin.ir import BlockRef, Imm

    if isinstance(op, Imm):
        text = f"{op.value}'{op.text}'" if op.text else str(op.value)
    elif isinstance(op, BlockRef):
        text = "@" + op.name
    else:
        text = repr(op)
    if mod & 2:
        text = f"|{text}|"
    return "-" + text if mod & 1 else text


def _reg_text(b, r) -> str:
    """A native register tuple ("reg", value, file, lo, n) as its VReg repr."""
    from thin.ir import VReg

    _, value, _, lo, n = r
    return (
        repr(VReg(b.values[value], lo, n))
        if value >= 0
        else f"s:_lit[0]x{n}" + (f"+{lo}" if lo else "")
    )


def _attrs(b, i) -> list:
    from thin import ir

    a = i.attrs or {}
    out = []
    if ir.Comment in a:
        out.append(f'comment="{a[ir.Comment].text}"')
    if ir.Tag in a:
        out.append(f"tag={a[ir.Tag].value!r}")
    if ir.Ablation in a:
        out.append(f"abl={sorted(a[ir.Ablation].classes)!r}")
    if ir.AliasOk in a:
        out.append(f"alias_ok={sorted(a[ir.AliasOk].pairs, key=repr)!r}")
    if ir.Staged in a:
        out.append(f"staged={_staged(b, a[ir.Staged])}")
    if ir.WaitUses in a:
        out.append(f"wait_uses={_wait_uses(b, a[ir.WaitUses])}")
    if ir.InRegion in a:
        out.append(f"in_region={a[ir.InRegion].id}")
    if ir.Places in a:
        out.append(f"places={list(a[ir.Places].names)!r}")
    if ir.Published in a:
        out.append(f'published="{a[ir.Published].why}"')
    if ir.LdsAccess in a:
        out.append(f"lds={_lds(b, a[ir.LdsAccess])}")
    if ir.M0Set in a:
        out.append(f"m0_set={a[ir.M0Set].value}")
    if ir.WavesTaken in a:
        out.append(f"waves_taken={sorted(a[ir.WavesTaken].waves)}")
    if ir.Fence in a:
        out.append("fence")
    return out


def _staged(b, s) -> str:
    stage = s.stage if hasattr(s, "stage") else s
    if stage[0] == "add":
        return repr(("add", _reg_text(b, stage[1]), stage[2]))
    return repr(tuple(stage))


def _wait_uses(b, w) -> str:
    regs = w.regs if hasattr(w, "regs") else w
    return "[" + ", ".join(r if isinstance(r, str) else _reg_text(b, r) for r in regs) + "]"


def _lds(b, x) -> str:
    if hasattr(x, "what"):
        region, what, field, var, ordered = (
            x.region,
            x.what,
            x.field,
            x.offset is not None,
            x.ordered,
        )
        name = getattr(region, "name", region)
    else:
        region, what, field, var, ordered = x
        name = _region_name(b, region)
    text = f'({name}, "{what}", field={field}, offset={"var" if var else "none"}'
    return text + (f', ordered="{ordered}")' if ordered else ")")


def _region_name(b, k) -> str:
    from thin.dsl import Region

    return Region(b.layout, k).name


def built(b) -> list:
    """The stream before any pass, in the DSL vectors' notation."""
    return [built_line(b, i) for i in b.ins]


def values(b) -> list:
    """The values, in the DSL vectors' notation."""
    out = []
    for v in b.values:
        line = f"{v.id} {v.file} {v.name} n={v.n} align={v.align}"
        line += f" index={v.index}" if v.index else ""
        line += f" pin={v.pin}" if v.pin is not None else ""
        line += f" same_as={v.same_as.id}" if v.same_as is not None else ""
        out.append(line)
    return out


def _line(b, i) -> str:
    if i is None:
        return "None"
    return built_line(b, i)


def built_line(b, i) -> str:
    """One instruction's line of the built notation."""
    mods = i.mods or bytes(len(i.ops))
    ops = ", ".join(_operand(op, mods[k] if k < len(mods) else 0) for k, op in enumerate(i.ops))
    fields = ", ".join(f"{k}={v}" for k, v in i.fields.items())
    line = i.desc.name + (" " + ops if ops else "") + (" {" + fields + "}" if fields else "")
    attrs = _attrs(b, i)
    return line + (" ; " + " ".join(attrs) if attrs else "")


def records(b) -> dict:
    """The program's records, in the DSL vectors' notation."""
    b.ins  # noqa: B018 (syncs the records)
    out: dict = {}
    p = b.program
    if p.regions:
        out["regions"] = [
            f"id={r.id} entry=`{_line(b, r.entry)}` save={r.save!r} checked={r.checked}"
            for r in p.regions
        ]
    if p.loops:
        out["loops"] = [
            f"{x.stem}: `{_line(b, x.first)}` .. `{_line(b, x.last)}` conservative={x.conservative}"
            for x in p.loops
        ]
    if b.deferred:
        out["deferred"] = [kind for kind, _ in b.deferred]
    for f in "vs":
        rf = b.files[f]
        for key, names in (
            ("entered", rf.entered),
            ("pinned", rf.pinned),
            ("lanes_covered", rf.lanes),
        ):
            if names:
                out[f"{f}.{key}"] = list(names)
    if b.needs:
        out["needs"] = b.needs
    if b.layout is not None:
        out["lds_table"] = b.layout.table()
    if b.wave_reg is not None:
        out["wave_index"] = f"{b.wave_reg!r} nwaves={b.nwaves} by_wave={b.by_wave}"
    return out


def array(value):
    """A value of the opspec notation: {np: dtype, shape, strided?} a
    zeroed numpy array (strided: every other element of one twice as long); {torch:
    dtype, shape} a zeroed CPU tensor; {np_int: v} a numpy int64; {list: n} n floats;
    {float: v}; any other JSON value as itself."""
    import numpy as np

    if not isinstance(value, dict):
        return value
    if "np" in value:
        shape = tuple(value["shape"])
        if value.get("strided"):
            return np.zeros((*shape[:-1], shape[-1] * 2), dtype=value["np"])[..., ::2]
        return np.zeros(shape, dtype=value["np"])
    if "torch" in value:
        import torch

        return torch.zeros(tuple(value["shape"]), dtype=getattr(torch, value["torch"]))
    if "np_int" in value:
        return np.int64(value["np_int"])
    if "list" in value:
        return [0.0] * value["list"]
    if "float" in value:
        return float(value["float"])
    return value


def compile_case(case: dict):
    """A codegen/kernels.json case through the DSL and the pipeline.

    ``derive`` runs ``src`` on each Builder it makes, then ``compile_kernel``
    renders the placed Builder with the case's symbol, wgsize and max_grid;
    the Builder and the Artifact are returned.
    """
    from thin import dsl

    b = dsl.derive(lambda: run_snippet(case["src"]))
    art = dsl.compile_kernel(b, case["symbol"], wgsize=case["wgsize"], max_grid=case["max_grid"])
    return b, art


def run_call(case: dict) -> dict:
    """Runs a codegen/refusals.json case's ``call``.

    The statements run with the language's names, ``using`` and ``build``
    (a fresh Builder with ``src`` run on it) in scope; the scope after them
    is returned, so the caller reads ``art`` or evaluates ``ok_of`` in it.
    """
    from thin import codegen, dsl

    scope = {name: getattr(dsl, name) for name in dsl.__all__}
    scope.update(using=codegen.using, build=lambda: run_snippet(case["src"]))
    exec(case["call"], scope)
    return scope
