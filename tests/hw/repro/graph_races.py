"""Stress ROCm 7.1 with twenty graph-path matrix races in one process.
Usage: python graph_races.py. Repeated launches can fail after about 2^14 calls."""

import os

import torch

import metron

torch.cuda.set_device(0)
n = 256
a, b = torch.randn(n, n), torch.randn(n, n)
p = metron.Point("matmul 256", {"a": a, "b": b}, {"c": torch.empty(n, n, device="meta")}, None)


def mm(x):
    torch.matmul(x.a, x.b, out=x.c)


for i in range(20):
    r = metron.race([metron.torch_arm("a", mm), metron.torch_arm("b", mm)], p, budget_s=1)
    assert r.validators["launches"] == "graph", r.validators
    print(i, r.rounds, round(r.versus("b", "a").ratio, 4), flush=True)
torch.cuda.synchronize()
# marker; os._exit skips torch's graph teardown, which segfaults alone.
print("races done 20", flush=True)
os._exit(0)
