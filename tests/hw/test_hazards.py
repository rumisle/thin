"""Exercise gfx1201 hazard chains and negative controls on the GPU."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from tests.hw.test_run import KERNELS, _case_args, _poison, kernel_of

pytestmark = pytest.mark.gfx1201
torch = pytest.importorskip("torch")

STEPS = 48
WAVES = 2  # wgsize 64 on a wave32 target: two waves a workgroup
GROUPS = 8


@pytest.fixture(autouse=True)
def no_store():
    import thin

    before = thin.use_store(None)
    yield
    thin.use_store(before)


def chain_kernel(seed: int, steps: int = STEPS):
    """A KernelDef of `steps` random dependent ops a lane; out[lane * steps + j]: step j."""
    from thin import KernelDef
    from thin.dsl import Args, O, cvt_f32, cvt_u32, exp2, first_lane, log2, ptr, rcp, rsq, sqrt

    rng = np.random.default_rng(seed)

    def body(b, k):
        V, S = b.v, b.s  # noqa: N806 - the DSL's spelling
        x, out = k.load()
        i = V.u32(k.wgid * 64 + k.tid, name="i")
        a = V.u32(name="a")
        b.emit(O.global_load(a, V.u32(i * 4, name="xo"), x))
        b.wait(a, "x landed")
        base = V.u32(i * (steps * 4), name="base")
        pools = {"u": [a], "f": [], "p": []}

        def pick(kind):
            pool = pools[kind]
            return pool[-1 - int(rng.integers(0, min(4, len(pool))))]

        for j in range(steps):
            ops = ["add", "xor", "shl", "cvtf", "exp", "sgpr"]
            if pools["f"]:
                ops += ["fadd", "fmul", "cvtu"]
            if pools["p"]:
                ops += ["log", "rcp", "sqrt", "rsq", "pmul"]
            op = ops[int(rng.integers(0, len(ops)))]
            n = f"r{j}"
            if op == "add":
                r, kind = V.u32(pick("u") + pick("u"), name=n), "u"
            elif op == "xor":
                r, kind = V.u32(pick("u") ^ (pick("u") + j), name=n), "u"
            elif op == "shl":
                r, kind = V.u32(pick("u") << (pick("u") & 7), name=n), "u"
            elif op == "cvtf":
                r, kind = V.f32(cvt_f32(pick("u") & 1023), name=n), "f"
            elif op == "fadd":
                r, kind = V.f32(pick("f") + pick("f"), name=n), "f"
            elif op == "fmul":
                r, kind = V.f32(pick("f") * 0.5, name=n), "f"
            elif op == "cvtu":
                r, kind = V.u32(cvt_u32(pick("f")), name=n), "u"
            elif op == "exp":
                e = V.f32(cvt_f32(pick("u") & 31), name=f"{n}e")
                r, kind = V.f32(exp2(e - 15.0), name=n), "p"  # 2^-15 .. 2^16, exact
            elif op == "log":
                r, kind = V.f32(log2(pick("p")), name=n), "f"  # an integer, exact
            elif op == "rcp":
                r, kind = V.f32(rcp(pick("p")), name=n), "p"
            elif op == "pmul":
                r, kind = V.f32(pick("p") * 2.0, name=n), "p"
            elif op in ("sqrt", "rsq"):
                p = pick("p")
                sq = V.f32(p * p, name=f"{n}q")  # a square of a power of two
                r = V.f32(sqrt(sq) if op == "sqrt" else rsq(sq), name=n)
                kind = "p"
            else:  # sgpr: VALU writes an SGPR, the SALU reads and rewrites it, a VALU reads it
                s = S.u32(first_lane(pick("u")), name=f"{n}s")
                t = S.u32(s + (j + 1), name=f"{n}t")
                u = V.u32(pick("u") + t, name=f"{n}u")
                t[...] = t ^ 0x55
                r, kind = V.u32(u + t, name=n), "u"
            pools[kind].append(r)
            b.emit(O.global_store(base, r, out, offset=4 * j or None))

    args = Args(x=ptr, out=ptr)
    return KernelDef(f"chain{seed}", args, body, wgsize=64, max_grid=(GROUPS,))


def _inputs(seed):
    rng = np.random.default_rng(1000 + seed)
    n = 64 * GROUPS
    return {"x": rng.integers(0, 2**32, n, dtype=np.uint32), "out": _poison(n * STEPS)}


def _hazard_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.rstrip().endswith("// hazard")
            or "// hazard:" in line]  # fmt: skip


def strip_hazards(text: str) -> str:
    """The .s without the instructions the hazard pass placed."""
    drop = set(_hazard_lines(text))
    return "\n".join(line for line in text.splitlines() if line not in drop) + "\n"


def _stripped_kernel(k):
    from thin import codegen
    from thin.runtime.kernel import Kernel

    text = strip_hazards(k.asm)
    co = codegen.code_object(text)
    return Kernel(k.name, text, k.manifest, co, k.params, k.address + ":stripped", k.definition)


def _gpu(k, grid, args, outs):
    on = {n: torch.from_numpy(v.copy()).cuda() if isinstance(v, np.ndarray) else v
          for n, v in args.items()}  # fmt: skip
    k.bind(grid, on).run()
    torch.cuda.synchronize()
    return {n: on[n].cpu().numpy() for n in outs}


def _emulated(k, grid, args, outs):
    a = {n: v.copy() if isinstance(v, np.ndarray) else v for n, v in args.items()}
    got = k.bind(grid, a).emulate()
    return {n: np.asarray(got.get(n, a[n])) for n in outs}


SEEDS = list(range(12))


@pytest.mark.parametrize("seed", SEEDS)
def test_a_chain_runs_as_it_emulates(seed):
    import thin

    k = thin.compile(chain_kernel, seed=seed)
    placed = _hazard_lines(k.asm)
    assert any("s_delay_alu" in line for line in placed)
    args = _inputs(seed)
    emu, gpu = _emulated(k, (GROUPS,), args, ["out"]), _gpu(k, (GROUPS,), args, ["out"])
    wrong = np.flatnonzero(emu["out"] != gpu["out"])
    assert not len(wrong), (
        f"{len(wrong)} words differ; first at lane {wrong[0] // STEPS}, step {wrong[0] % STEPS}"
    )


def _hints(text):
    """The kinds of hazard instruction a text has, counted."""
    out = {}
    for line in _hazard_lines(text):
        op = line.split()[0]
        key = op if op != "s_delay_alu" else "s_delay_alu " + line.split("hazard: ")[-1].strip()
        out[key] = out.get(key, 0) + 1
    return out


RUNS = int(os.environ.get("THIN_HAZARD_RUNS", "20"))


def test_the_negative_control_strips_every_hazard_instruction(tmp_path):
    """Records whether code without the pass's instructions ever computes otherwise."""
    import thin

    report = {"runs": RUNS, "chains": {}, "kernels": {}}
    for seed in SEEDS:
        k = thin.compile(chain_kernel, seed=seed)
        bare = _stripped_kernel(k)
        assert not _hazard_lines(bare.asm)
        args = _inputs(seed)
        want = _emulated(k, (GROUPS,), args, ["out"])["out"]
        bad = [
            int((_gpu(bare, (GROUPS,), args, ["out"])["out"] != want).sum()) for _ in range(RUNS)
        ]
        report["chains"][seed] = {"placed": _hints(k.asm), "words_wrong_per_run": bad}
    for name in (
        "guarded_permute_b64",
        "matrix_recurrence_b32",
        "scalar_reuse_b64",
        "loop_filter_b64",
    ):
        k = kernel_of(KERNELS[name])
        bare = _stripped_kernel(k)
        grid, args, outs = _case_args(name, np.random.default_rng(1))
        want = _emulated(k, grid, args, outs)
        if name == "guarded_permute_b64":
            want["out"][13] = np.uint32(0)  # undefined (test_run): never compared
        bad = []
        for _ in range(RUNS):
            got = _gpu(bare, grid, args, outs)
            if name == "guarded_permute_b64":
                got["out"][13] = np.uint32(0)
            bad.append(
                sum(int((got[n].view(np.uint8) != want[n].view(np.uint8)).sum()) for n in outs)
            )
        report["kernels"][name] = {"placed": _hints(k.asm), "bytes_wrong_per_run": bad}
    path = os.environ.get("THIN_HW_REPORT_DIR", str(tmp_path))
    with open(os.path.join(path, "hazard_control.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report))
    # Measured: the v_nop between two dependent WMMAs is needed - the second
    # reads a stale D on every run without it. The rest, stripped, computed the same.
    assert all(n > 0 for n in report["kernels"]["matrix_recurrence_b32"]["bytes_wrong_per_run"])
