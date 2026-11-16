"""Emulator numerics, bank-conflict, and manifest tests for `examples.gfx942.gemm_bf16`."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import thin
from examples.gfx942.gemm_bf16 import build_gemm_bf16, default_gemm_config
from thin.target import targeting


def _f32_to_bf16_u16(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.float32).view(np.uint32)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)


def _bf16_u16_to_f32(u: np.ndarray) -> np.ndarray:
    return (u.astype(np.uint32) << 16).view(np.float32)


def test_default_gemm_configs_compile_cleanly():
    """All four square MI300X configurations compile at 1 WG/CU with no warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with targeting("gfx942"):
            before = thin.use_store(None)
            try:
                for m in (2048, 4096, 8192, 16384):
                    kern = thin.compile(build_gemm_bf16, M=m)
                    assert kern.manifest.vgpr <= 256
                    assert kern.manifest.agpr <= 256
                    assert kern.manifest.lds <= 65536
                    assert kern.manifest.resident == 1
                    assert default_gemm_config(m)["BM"] in (64, 112, 144, 160)
            finally:
                thin.use_store(before)


@pytest.mark.parametrize(
    ("m", "n", "k", "cfg"),
    [
        (128, 256, 256, {"BM": 64, "BN": 256, "GROUP_M": 2}),
        (200, 512, 256, {"BM": 112, "BN": 512, "GROUP_M": 6}),
        (144, 512, 128, {"BM": 144, "BN": 512, "GROUP_M": 6}),
        (160, 512, 128, {"BM": 160, "BN": 512, "GROUP_M": 8}),
    ],
)
def test_gemm_bf16_emulator_numerics_and_zero_bank_conflicts(
    m: int, n: int, k: int, cfg: dict[str, int]
) -> None:
    """Checks numerical accuracy, M-tail clamping, and zero LDS bank conflicts on `gfx942`."""
    rng = np.random.default_rng(942 + m + n)
    a_u16 = _f32_to_bf16_u16(rng.standard_normal((m, k)) * 0.25)
    b_u16 = _f32_to_bf16_u16(rng.standard_normal((k, n)) * 0.25)
    c_u16 = np.zeros((m, n), dtype=np.uint16)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with targeting("gfx942"):
            before = thin.use_store(None)
            try:
                kern = thin.compile(build_gemm_bf16, M=m, N=n, K=k, **cfg)
                kern(a=a_u16, b=b_u16, c=c_u16).emulate()
            finally:
                thin.use_store(before)

    ref = _bf16_u16_to_f32(a_u16) @ _bf16_u16_to_f32(b_u16)
    got = _bf16_u16_to_f32(c_u16)
    assert np.allclose(got, ref, rtol=2e-2, atol=2e-2)
