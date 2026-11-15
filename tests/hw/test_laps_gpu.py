"""laps on gfx1201: the cycle counter and the constant clock read, kept and decoded."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gfx1201
torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def no_store():
    import thin

    before = thin.use_store(None)
    yield
    thin.use_store(before)


@pytest.mark.parametrize("keep", ["last", "first", "sum"])
def test_laps_record_and_decode_on_the_gpu(keep):
    """Two rounds a wave; the core clock against the 100 MHz one, pooled, is a GPU's clock."""
    import thin
    from tests.laps_kernels import gen_laps, laps_decl

    decl = laps_decl(keep)
    k = thin.compile(gen_laps, keep=keep)
    out = torch.zeros(64 * 2, dtype=torch.int32, device="cuda")
    bound = k.bind((2,), {"out": out, **decl.sizes(2)})
    bound.run()
    torch.cuda.synchronize()
    assert out.cpu().numpy().tolist() == [4 * t for t in range(128)]
    t = decl.decode(bound.owned["laps"], 2)
    assert t.rounds == [[2, 2], [2, 2]]
    mhz = [m for row in t.clock_mhz for m in row]
    assert all(m is None or m > 0 for m in mhz), mhz
    assert 300 < t.clock < 5000, t.summary()  # pooled over the waves' spans
    if keep == "sum":
        assert all(v > 0 for n in decl.names for row in t.laps[n] for v in row)
    else:
        phases = t.phases()
        assert len(phases["store"]) == 8 and all(0 < p < 1 << 24 for p in phases["store"])
    assert "never taken" not in t.summary()
    assert isinstance(t.summary(), str) and np.isfinite(t.clock)
