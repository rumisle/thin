"""The operands of shipped kernels (see docs/dev/lift.md) on the emulator:
vcc_hi on wave32 and a null buffer soffset. The ops' semantics are
alu.json's derived cases."""

import numpy as np
import pytest

HEAD = "\ts_load_b64 s[4:5], s[0:1], 0x0\n\ts_wait_kmcnt 0\n\tv_lshlrev_b32 v1, 2, v0\n"
TAIL = "\tglobal_store_b32 v1, v2, s[4:5]\n\ts_endpgm\n"


def _run(body, x=None):
    """One wave of HEAD + body + TAIL (s[4:5] the output y, x's base in s[6:7]);
    y's dwords."""
    from thin.emu import Cluster, Memory

    m = Memory()
    y = m.alloc("y", 128)
    xs = m.alloc("x", 128)
    xs.write(0, np.asarray(x if x is not None else np.zeros(32), np.uint32).tobytes())
    k = m.alloc("k", 16, tagged=False)
    k.write(0, np.array([y.base, xs.base], dtype=np.uint64).tobytes())
    text = HEAD + "\ts_load_b64 s[6:7], s[0:1], 0x8\n\ts_wait_kmcnt 0\n" + body + TAIL
    Cluster(text, {"wgsize": 32, "vgpr": 8}, m, [k]).run()
    return np.frombuffer(y.data, np.uint32)


def test_vcc_hi_is_a_scalar_apart_from_the_wave32_mask():
    body = (
        "\ts_mov_b32 vcc_lo, 0x55\n\ts_mov_b32 vcc_hi, 7\n\ts_mov_b32 vcc_hi, 0\n"
        "\ts_mov_b32 s8, vcc_hi\n\ts_add_co_u32 s8, s8, vcc_lo\n\tv_mov_b32 v2, s8\n"
    )
    assert set(_run(body).tolist()) == {0x55}  # vcc_hi 0, the mask unchanged
    masked = "\ts_mov_b32 vcc_lo, 0\n\ts_mov_b32 vcc_hi, -1\n\tv_cndmask_b32 v2, 1, 2, vcc_lo\n"
    assert set(_run(masked).tolist()) == {1}  # no lane of the mask is vcc_hi's


def test_vcc_hi_as_a_lane_mask_is_refused():
    from thin.emu import EmuError

    body = "\ts_mov_b32 vcc_hi, 0\n\tv_cndmask_b32 v2, 1, 2, vcc_hi\n"
    with pytest.raises(
        EmuError, match=r"^v_cndmask_b32 v2, 1, 2, vcc_hi: vcc_hi is not a lane mask on wave32$"
    ):
        _run(body)


@pytest.mark.parametrize("soffset", ["null", "off"])
def test_a_null_buffer_soffset_reads_zero_as_off(soffset):
    x = np.arange(32, dtype=np.uint32) * 3 + 1
    body = (
        "\ts_mov_b32 s8, s6\n\ts_and_b32 s9, s7, 0xffff\n\ts_mov_b32 s10, 128\n"
        "\ts_mov_b32 s11, 0\n"
        f"\tbuffer_load_b32 v2, v1, s[8:11], {soffset} offen\n\ts_wait_loadcnt 0\n"
    )
    assert np.array_equal(_run(body, x), x)
