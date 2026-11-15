"""The environment handed to child processes."""

from __future__ import annotations

import os

__all__ = ["HASH_SEED", "HOLD", "child", "gpu_hold"]

HOLD = "NM_GPU_EXCLUSIVE"
"""The GPU service's hold variable."""

_PASSED = (
    "PATH",
    "HOME",
    "LANG",
    "TMPDIR",
    "LD_LIBRARY_PATH",
    "ROCM_PATH",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    HOLD,
)

HASH_SEED = {"PYTHONHASHSEED": "12345"}
"""The determinism test's second process's hash seed."""

_NAMED = (HASH_SEED,)


def child(*named) -> dict[str, str]:
    """A child's environment: the passed variables as set here, then the named constants."""
    env = {k: os.environ[k] for k in _PASSED if k in os.environ}
    for n in named:
        if not any(n is c for c in _NAMED):
            raise ValueError("child: only the named constants")
        env.update(n)
    return env


def gpu_hold() -> str:
    """``exclusive``, ``shared`` or ``unknown``, from the hold variable."""
    return {"1": "exclusive", "0": "shared"}.get(os.environ.get(HOLD, ""), "unknown")
