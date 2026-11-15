"""Stress ROCm 7.1 cumulative graph replay.
Usage: python graph_replays.py K R; capture K calls and replay R times."""

import os
import sys

import torch

torch.cuda.set_device(0)
K, R, n = int(sys.argv[1]), int(sys.argv[2]), 256
stride = 4 * n * n + 4096
bufs = {k: torch.full((K * stride,), 7, dtype=torch.uint8, device="cuda") for k in "abc"}


def v(k, i):
    return bufs[k][i * stride : i * stride + 4 * n * n].view(torch.float32).view(n, n)


torch.matmul(v("a", 0), v("b", 0), out=v("c", 0))
torch.cuda.synchronize()
s = torch.cuda.Stream()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, stream=s):
    for i in range(K):
        torch.matmul(v("a", i), v("b", i), out=v("c", i))
for _ in range(R):
    with torch.cuda.stream(s):
        g.replay()
    s.synchronize()
s.synchronize()
# marker; os._exit skips torch's graph teardown, which segfaults alone.
print(f"replays done {K * R}", flush=True)
os._exit(0)
