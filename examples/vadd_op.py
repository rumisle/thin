"""An example operation: example.vadd and its implementation add."""

from __future__ import annotations

from typing import ClassVar

from examples.vadd import build_vadd

N_MAX = (1 << 30) - 1
"""The largest n the implementation serves."""


def _spec():
    from thin.opspec import In, Int, Out, Spec

    class VAdd(Spec, name="example.vadd"):
        """z = x + y over n f32 elements."""

        n = Int(lo=1)
        dists: ClassVar[dict] = {"scale": [1.0, 1e30]}

        def signature(op):  # noqa: N805 - the spelling
            return {"x": In("f32", op.n), "y": In("f32", op.n), "z": Out("f32", op.n)}

        def inputs(op, seed, *, scale=1.0):  # noqa: N805
            import numpy as np

            rng = np.random.default_rng(seed)
            x = (rng.standard_normal(op.n) * scale).astype(np.float32)
            y = (rng.standard_normal(op.n) * scale).astype(np.float32)
            return {"x": x, "y": y}

        def check(op, inputs, outputs):  # noqa: N805
            import numpy as np

            want = inputs["x"] + inputs["y"]
            got = outputs["z"]
            wrong = np.flatnonzero(got.view(np.uint32) != want.view(np.uint32))
            if not len(wrong):
                return None
            i = int(wrong[0])
            return f"z[{i}] = {float(got[i])}, want {float(want[i])} ({len(wrong)} wrong)"

    return VAdd


VAdd = _spec()


def _register():
    from thin import compile
    from thin.opspec import implements

    @implements(VAdd, tunables={"BLOCK": (256, 512)}, serves=lambda op, **_: {"n": (1, N_MAX)})
    def add(op, BLOCK):  # noqa: N803 - a tunable
        k = compile(build_vadd, BLOCK=BLOCK, n=op.n.range)
        return k(x=op.x, y=op.y, z=op.z, n=op.n)

    return add


add = _register()
