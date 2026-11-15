"""Stress ROCm 7.1 graph capture with repeated kernel nodes.
Usage: python graph_nodes.py K; K is the number of captured nodes."""

import sys

import torch

torch.cuda.set_device(0)
K, n, G = int(sys.argv[1]), 256, int(sys.argv[2])
stride = 4 * n * n + 4096
bufs = {k: torch.full((K * stride,), 7, dtype=torch.uint8, device="cuda") for k in "abc"}


def v(k, i):
    return bufs[k][i * stride : i * stride + 4 * n * n].view(torch.float32).view(n, n)


s = torch.cuda.Stream()
torch.matmul(v("a", 0), v("b", 0), out=v("c", 0))
torch.cuda.synchronize()
gs = []
for j in range(G):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for i in range(K):
            torch.matmul(v("a", i), v("b", i), out=v("c", i))
    gs.append(g)
    print("captured", j, flush=True)
    for _ in range(2):
        with torch.cuda.stream(s):
            g.replay()
    torch.cuda.synchronize()
    print("replayed", j, flush=True)
print("ok")
