# thin

**What you write is what you get.**

thin is a Python DSL for writing AMD GPU kernels at the instruction level, with an emulator that models the hardware exactly.

thin currently supports gfx1201 (RDNA 4).

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

`emulate()` runs the kernel on the CPU, on numpy arrays. On a gfx1201 GPU,
bind torch tensors and call `run()`:

```python
import torch

xg, yg = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
zg = torch.zeros_like(xg)
kernel.bind(grid=cdiv(n, 256), args={"x": xg, "y": yg, "z": zg, "n": n}).run()
assert torch.equal(zg, xg + yg)
```

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

The first build compiles the Rust core; after that, a Rust edit rebuilds
the extension in about a second (with pip, run `pip install -e .` again).
Hardware tests need a gfx1201 GPU: `pytest -m gfx1201`. The timing
harness, `metron`, lives in `python/metron`.

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
