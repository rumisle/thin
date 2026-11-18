# metron

A timing and correctness harness for GPU kernels. Candidates compute the same
outputs for one input point, are checked before timing, then run in shuffled
rounds until comparisons against arm 0 settle or the budget is spent.

Input restores, launch mode and correctness checks are explicit. A host backend
supports deterministic protocol tests without a GPU. Affected ROCm stacks use
host launches rather than unsafe graph replay.

metron is kept in the thin repository and is not published to PyPI. Install
from a checkout with `pip install -e python/metron`; install thin separately
for `metron.thin` and `metron.maker`. Supply the torch build for your GPU.

See `docs/dev/ARCHITECTURE.md`, `docs/dev/hardware.md` and `docs/dev/limitations.md` in the
repository. Licensed under Apache-2.0; see LICENSE.
