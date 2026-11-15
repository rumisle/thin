"""Regression checks for wide constants and signed zero on gfx1201."""

from __future__ import annotations

import numpy as np
import pytest

from tests import dsl_edges as E  # noqa: N812 - the cases, as E

pytestmark = pytest.mark.gfx1201
torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def no_store():
    import thin

    before = thin.use_store(None)
    yield
    thin.use_store(before)


def _run(k, args, out):
    t = {n: torch.from_numpy(v).cuda() if isinstance(v, np.ndarray) else v for n, v in args.items()}
    k.bind((1,), t).run()
    torch.cuda.synchronize()
    return t[out].cpu().numpy()


@pytest.mark.parametrize("p", E.PS)
def test_u64_plus_minus_every_edge_constant_scalar(p):
    import thin

    k = thin.compile(E.gen_wide_scalar)
    got = _run(k, {"p": p, "out": np.zeros(E.SLOTS, np.int64)}, "out").view(np.uint64)
    assert got.tolist() == E.wide_want(p)


def test_u64_plus_minus_every_edge_constant_vector():
    import thin

    k = thin.compile(E.gen_wide_vector)
    ps = np.array(E.PS + [0] * (32 - len(E.PS)), np.uint64).view(np.int64)
    got = _run(k, {"ps": ps, "out": np.zeros(32 * E.SLOTS, np.int64)}, "out").view(np.uint64)
    for lane, p in enumerate(E.PS):
        assert got[lane * E.SLOTS : (lane + 1) * E.SLOTS].tolist() == E.wide_want(p), hex(p)


def test_the_sign_of_zero():
    import thin

    k = thin.compile(E.gen_zero)
    src = np.zeros(32, np.float32)
    src[: len(E.ZERO_INPUTS)] = E.ZERO_INPUTS
    got = _run(k, {"src": src, "out": np.zeros(32 * 6, np.int32)}, "out").view(np.uint32)
    want = E.zero_want()
    assert [hex(x) for x in got[: len(want)]] == [hex(x) for x in want]


def test_the_borrow_chain_runs_as_it_emulates():
    """on hardware: results and borrow masks equal to the emulation's
    (and the guide's), under a partial EXEC."""
    import thin

    k = thin.compile(E.gen_borrow)
    xs, ys = E.borrow_inputs()
    args = {"xs": xs, "ys": ys, "d": np.zeros(96, np.uint32), "masks": np.zeros(3, np.uint32)}
    emu = k.bind((1,), {n: v.copy() for n, v in args.items()}).emulate()
    gd = _run(
        k, {**args, "d": args["d"].view(np.int32), "masks": args["masks"].view(np.int32)}, "d"
    )
    t = {
        n: torch.from_numpy(v.view(np.int32) if v.dtype == np.uint32 else v).cuda()
        for n, v in args.items()
    }
    k.bind((1,), t).run()
    torch.cuda.synchronize()
    gm = t["masks"].cpu().numpy().view(np.uint32)
    gd = gd.view(np.uint32).reshape(3, 32)[:, : E.ACTIVE]
    ed = np.asarray(emu["d"]).reshape(3, 32)[:, : E.ACTIVE]
    assert gd.tolist() == ed.tolist()
    assert [hex(m) for m in gm] == [hex(m) for m in np.asarray(emu["masks"])]
    want_d, want_m = E.borrow_want()
    assert gd.tolist() == want_d.reshape(3, 32)[:, : E.ACTIVE].tolist()
    assert [hex(m) for m in gm] == [hex(m) for m in want_m]
