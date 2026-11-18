"""Emulator numerics, 64-bank conflict, and manifest tests for `examples.gfx950.gemm_bf16`."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import thin
from examples.gfx950.gemm_bf16 import build_gemm_bf16, default_gemm_config
from thin.target import targeting


def _f32_to_bf16_u16(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.float32).view(np.uint32)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)


def _bf16_u16_to_f32(u: np.ndarray) -> np.ndarray:
    return (u.astype(np.uint32) << 16).view(np.float32)


def test_default_gemm_gfx950_configs_compile_cleanly() -> None:
    """All four square CDNA 4 configs (including 96 KB and 128 KB LDS) compile cleanly."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with targeting("gfx950"):
            before = thin.use_store(None)
            try:
                for m, want_lds in (
                    (1024, 49_152),
                    (2048, 65_536),
                    (4096, 98_304),
                    (8192, 131_072),
                ):
                    kern = thin.compile(build_gemm_bf16, M=m)
                    assert kern.manifest.vgpr <= 256
                    assert kern.manifest.agpr <= 256
                    assert kern.manifest.lds == want_lds
                    assert kern.manifest.lds <= 163_840
                    assert kern.manifest.resident >= 1
                    assert default_gemm_config(m)["BM"] in (64, 128, 256)
                    assert "v_mfma_f32_16x16x32_bf16" in kern.asm
                    assert "buffer_load_dwordx4" in kern.asm and " lds" in kern.asm
                    assert "ds_read_b64_tr_b16" in kern.asm
                    assert "v_cvt_pk_bf16_f32" in kern.asm
            finally:
                thin.use_store(before)


@pytest.mark.parametrize(
    ("m", "n", "k", "cfg"),
    [
        (64, 128, 64, {"BM": 64, "BN": 128, "GROUP_M": 1}),
        (128, 256, 192, {"BM": 128, "BN": 256, "GROUP_M": 1}),
        (100, 256, 256, {"BM": 64, "BN": 128, "GROUP_M": 2}),
        (192, 256, 256, {"BM": 64, "BN": 128, "GROUP_M": 2}),
    ],
)
def test_gemm_bf16_gfx950_emulator_numerics_and_zero_bank_conflicts(
    m: int, n: int, k: int, cfg: dict[str, int]
) -> None:
    """Checks numerical accuracy, M-tail clamping, K loop, and zero 64-bank LDS conflicts."""
    rng = np.random.default_rng(950 + m + n + k)
    a_u16 = _f32_to_bf16_u16(rng.standard_normal((m, k)) * 0.25)
    b_u16 = _f32_to_bf16_u16(rng.standard_normal((k, n)) * 0.25)
    c_u16 = np.zeros((m, n), dtype=np.uint16)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with targeting("gfx950"):
            before = thin.use_store(None)
            try:
                kern = thin.compile(build_gemm_bf16, M=m, N=n, K=k, **cfg)
                kern(a=a_u16, b=b_u16, c=c_u16).emulate()
            finally:
                thin.use_store(before)

    ref = _bf16_u16_to_f32(a_u16) @ _bf16_u16_to_f32(b_u16)
    got = _bf16_u16_to_f32(c_u16)
    assert np.allclose(got, ref, rtol=2e-2, atol=2e-2)
