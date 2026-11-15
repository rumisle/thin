"""Stress ROCm 7.1 replay across two graphs or fresh captures.
Usage: python graph_replays_spread.py two|fresh K R."""

import os
import sys

import torch

torch.cuda.set_device(0)
mode, K, R, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), 256
stride = 4 * n * n + 4096
bufs = {k: torch.full((K * stride,), 7, dtype=torch.uint8, device="cuda") for k in "abc"}


def v(k, i):
    return bufs[k][i * stride : i * stride + 4 * n * n].view(torch.float32).view(n, n)


torch.matmul(v("a", 0), v("b", 0), out=v("c", 0))
torch.cuda.synchronize()
s = torch.cuda.Stream()


def make():
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for i in range(K):
            torch.matmul(v("a", i), v("b", i), out=v("c", i))
    return g


done = 0
if mode == "two":
    a, b = make(), make()
    for _ in range(R):
        for g in (a, b):
            with torch.cuda.stream(s):
                g.replay()
            s.synchronize()
            done += K
else:
    for _ in range(R):
        g = make()
        with torch.cuda.stream(s):
            g.replay()
        s.synchronize()
        done += K
        g.reset()
        del g
s.synchronize()
# marker; os._exit skips torch's graph teardown, which segfaults alone.
print(f"replays done {done}", flush=True)
os._exit(0)
