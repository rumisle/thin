# thin

**What you write is what you get.**

thin is a Python DSL for writing AMD GPU kernels at the instruction level, with an emulator that models the hardware exactly.

thin currently supports **gfx1201** (RDNA 4), **gfx942** (CDNA 3 / Instinct MI300X), and **gfx950** (CDNA 4 / Instinct MI350X & MI355X; offline-verified against AMD ISA XML, LLVM 22, and the CDNA 4 ISA manual, awaiting silicon confirmation).

## Install

```
pip install git+https://github.com/rumisle/thin
```

Building from source needs a Rust toolchain ([rustup](https://rustup.rs)).
The core needs only numpy; for torch tensors, install torch as well, with
the build you already use.

## Example

```python
import numpy as np
from thin import KernelDef, cdiv, compile
from thin.dsl import Args, O, ptr, u32


def body(b, k):
    V = b.v
    x, y, z, n = k.load()
    i = V.u32(k.wgid * 256 + k.tid)
    with b.lanes(i < n):
        off = V.u32(i * 4)
        a, c = V.f32(), V.f32()
        b.emit(O.global_load(a, off, x))
        b.emit(O.global_load(c, off, y))
        b.wait(a, c)
        a += c
        b.emit(O.global_store(off, a, z))


def vadd():
    args = Args(x=ptr, y=ptr, z=ptr, n=u32)
    return KernelDef("vadd", args, body, wgsize=256, max_grid=(1 << 20,))


n = 1000
x, y = np.random.rand(2, n).astype(np.float32)
z = np.zeros(n, np.float32)
kernel = compile(vadd)
kernel.bind(grid=cdiv(n, 256), args={"x": x, "y": y, "z": z, "n": n}).emulate()
assert np.array_equal(z, x + y)
```

`emulate()` runs the kernel on the CPU, on numpy arrays. On a supported AMD GPU
(`gfx1201`, `gfx942`, or `gfx950`), bind torch tensors and call `run()`:

```python
import torch

xg, yg = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
zg = torch.zeros_like(xg)
kernel.bind(grid=cdiv(n, 256), args={"x": xg, "y": yg, "z": zg, "n": n}).run()
assert torch.equal(zg, xg + yg)
```

Select a target explicitly with `thin.target.targeting` (default `gfx1201`):

```python
from thin.target import targeting

with targeting("gfx942"):
    mi300x_kernel = compile(vadd)
    mi300x_kernel.bind(grid=cdiv(n, 256), args={"x": x, "y": y, "z": z, "n": n}).emulate()
```

On `gfx942` (CDNA 3 / MI300X), thin models wave64 execution, AGPR (`b.a`) and
VGPR (`b.v`) accumulators, all dense `v_mfma_*` and sparse 4:2 `v_smfmac_*`
shapes (`b.mfma`, `b.smfmac`, `b.read_acc`, `b.write_acc`), pass-count-aware
`s_nop` matrix hazards, FNUZ FP8/BF8 (`e4m3fnuz` / `e5m2fnuz`), global/buffer
to LDS DMA (`O.global_load(tile, ...)` / `O.buffer_load(tile, ...)`), GFX9 DPP
broadcasts and wave shifts, packed `v_pk_*_f32` ops, and the 32-bank LDS
conflict model.

On `gfx950` (CDNA 4 / MI350X & MI355X), thin adds 128-bit `ENC_VOP3PX2`
encoding/decoding, doubled-K dense `v_mfma_*` and sparse `v_smfmac_*` shapes,
mixed-precision `v_mfma_f32_*_f8f6f4` (`b.mfma(..., fmt_a=..., fmt_b=...)`),
MX E8M0 block-scaled `v_mfma_scale_f32_*_f8f6f4` (`b.mfma_scale`), OCP MX
element formats (`e4m3fn`, `e5m2`, `e2m3`, `e3m2`, `e2m1`, `e8m0`),
`v_cvt_scalef32_*` / stochastic-rounding conversions / `v_prng_b32`, transpose
LDS reads (`O.ds_read_tr`), 96/128-bit direct-to-LDS DMA (`bytes=12 | 16`), the
160 KB / 64-bank LDS model, and CDNA 4 wait-state hazards. New CDNA 4 matrix
layouts and numerics are derived from the ISA manual (`provisional=True`) until
validated on MI350X/MI355X hardware.

## Development

With pip:

    python -m venv .venv && . .venv/bin/activate
    pip install -e '.[dev]' -e python/metron
    pytest
    cargo test --workspace

With uv:

    uv sync
    uv run pytest
    cargo test --workspace

The tests check thin's encodings and wait states against LLVM 22, so
`llvm-mc`, `llc` and `llvm-objcopy` from LLVM 22 must be on `PATH`; they
fail, rather than skip, without them.

The first build compiles the Rust core; after that, a Rust edit rebuilds
the extension in about a second (with pip, run `pip install -e .` again).
Hardware tests run per target: `pytest -m gfx1201` on RDNA 4,
`pytest -m gfx942` on Instinct MI300X, or `pytest -m gfx950` on Instinct
MI350X/MI355X. The timing harness, `metron`, lives in `python/metron`.

Documentation is in [docs/](https://github.com/rumisle/thin/tree/main/docs).

## Citation

Please cite this work as:

    rumisle, "thin: an instruction-level DSL for AMD GPU kernels", 2026.
    https://github.com/rumisle/thin

Or use the BibTeX citation:

```bibtex
@software{rumisle2026thin,
  author = {rumisle},
  title  = {thin: an instruction-level DSL for AMD GPU kernels},
  year   = {2026},
  url    = {https://github.com/rumisle/thin},
}
```

## License

thin is licensed under the [Functional Source License, Version 1.1, ALv2
Future License][license]. Third-party notices are in [NOTICE][notice].

[license]: https://github.com/rumisle/thin/blob/main/LICENSE.md
[notice]: https://github.com/rumisle/thin/blob/main/NOTICE
