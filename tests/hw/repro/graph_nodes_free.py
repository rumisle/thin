"""Stress ROCm 7.1 graph-node allocation while releasing each capture.
This distinguishes cumulative per-process limits from live graph size."""

import gc

import torch

torch.cuda.set_device(0)
K, n = 2000, 256
stride = 4 * n * n + 4096
bufs = {k: torch.full((K * stride,), 7, dtype=torch.uint8, device="cuda") for k in "abc"}


def v(k, i):
    return bufs[k][i * stride : i * stride + 4 * n * n].view(torch.float32).view(n, n)


s = torch.cuda.Stream()
for j in range(10):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for i in range(K):
            torch.add(v("a", i), v("b", i), out=v("c", i))
    with torch.cuda.stream(s):
        g.replay()
    torch.cuda.synchronize()
    del g
    gc.collect()
    torch.cuda.synchronize()
    print("round", j, "ok (", (j + 1) * K, "nodes made in all )", flush=True)
