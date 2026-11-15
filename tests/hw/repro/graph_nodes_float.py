"""Stress ROCm 7.1 graph capture using floating-point adds.
This probes cumulative launch limits without a custom code object."""

import sys

import torch

torch.cuda.set_device(0)
K, G, kind = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
n = 256
if kind == "floats":  # a fresh float tensor for every operand
    ts = [[torch.randn(n, n, device="cuda") for _ in range(3)] for _ in range(K)]

    def v(k, i):
        return ts[i]["abc".index(k)]
else:  # one float buffer, viewed (no u8)
    stride = n * n + 1024
    bufs = {k: torch.zeros(K * stride, device="cuda") for k in "abc"}

    def v(k, i):
        return bufs[k][i * stride : i * stride + n * n].view(n, n)


s = torch.cuda.Stream()
torch.cuda.synchronize()
gs = []
for j in range(G):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for i in range(K):
            torch.add(v("a", i), v("b", i), out=v("c", i))
    gs.append(g)
    with torch.cuda.stream(s):
        g.replay()
    torch.cuda.synchronize()
    print("replayed", j, flush=True)
print("ok")
