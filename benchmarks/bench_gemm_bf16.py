"""Square bf16 GEMM (`C = A @ B`) benchmark and correctness check on MI300X (`gfx942`)."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
import subprocess
import threading
import time

import numpy as np
import torch

import thin
from examples.gfx942.gemm_bf16 import build_gemm_bf16, default_gemm_config
from thin.target import targeting

DATASHEET_PEAK_TFLOPS = 1307.44
MEASURED_ROOFLINE_TFLOPS = 1132.98
TUNED_HIPBLASLT_TFLOPS = {
    2048: 443.6,
    4096: 650.2,
    8192: 693.9,
    16384: 650.1,
}


def sample_smi_once() -> dict[str, float | int | None]:
    """Reads socket power (W) and mean XCD GFX clock (MHz) from `amd-smi`."""
    out = subprocess.check_output(["amd-smi", "metric", "-c", "-p"], text=True, timeout=5)
    power_m = re.search(r"SOCKET_POWER:\s*(\d+)\s*W", out)
    gfx_clks = [int(x) for x in re.findall(r"GFX_\d+:\s+CLK:\s*(\d+)\s*MHz", out)]
    return {
        "power_w": int(power_m.group(1)) if power_m else None,
        "gfx_mean_mhz": round(float(np.mean(gfx_clks)), 1) if gfx_clks else None,
    }


def time_events(
    fn, warmup: int = 25, iters: int = 60, min_warmup_ms: float = 30.0
) -> tuple[float, float, float]:
    """Returns `(median_ms, p10_ms, p90_ms)` across `iters` per-call HIP event pairs."""
    probe_s = torch.cuda.Event(enable_timing=True)
    probe_e = torch.cuda.Event(enable_timing=True)
    fn()
    probe_s.record()
    for _ in range(4):
        fn()
    probe_e.record()
    torch.cuda.synchronize()
    est_ms = max(0.005, probe_s.elapsed_time(probe_e) / 4.0)
    eff_warmup = max(warmup, math.ceil(min_warmup_ms / est_ms))
    for _ in range(eff_warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends, strict=True))
    med = ms[iters // 2] if iters % 2 == 1 else 0.5 * (ms[iters // 2 - 1] + ms[iters // 2])
    return float(med), float(ms[iters // 10]), float(ms[(iters * 9) // 10])


def measure_sustained_telemetry(fn, duration_s: float = 0.65) -> tuple[float | None, float | None]:
    """Samples socket power and XCD clock in a separate untimed pass."""
    samples: list[dict] = []
    stop = [False]

    def sampler():
        time.sleep(0.12)
        while not stop[0]:
            with contextlib.suppress(Exception):
                samples.append(sample_smi_once())
            time.sleep(0.12)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) < duration_s:
        for _ in range(8):
            fn()
        torch.cuda.synchronize()
    stop[0] = True
    th.join(timeout=2.0)
    active = [s for s in samples if s.get("power_w") and s["power_w"] > 220]
    pwr = round(float(np.mean([s["power_w"] for s in active])), 1) if active else None
    clk = round(float(np.mean([s["gfx_mean_mhz"] for s in active])), 1) if active else None
    return pwr, clk


def benchmark_config(
    m: int,
    cfg: dict,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    ref: torch.Tensor,
    *,
    warmup: int = 25,
    iters: int = 60,
) -> dict:
    """Checks `cfg` against `ref` and measures HIP-event timing and sustained telemetry."""
    kern = thin.compile(build_gemm_bf16, M=m, N=m, K=m, **cfg)
    grid = (thin.cdiv(m, cfg["BM"]) * (m // cfg["BN"]),)
    bound = kern.bind(
        grid, {"a": a.view(torch.uint16), "b": b.view(torch.uint16), "c": c.view(torch.uint16)}
    )

    c.zero_()
    bound.run()
    torch.cuda.synchronize()

    diff = (c.float() - ref.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    assert torch.allclose(c.float(), ref.float(), rtol=2e-2, atol=2e-2), f"M={m}: max_abs={max_abs}"

    med_ms, p10_ms, p90_ms = time_events(bound.run, warmup=warmup, iters=iters)
    assert torch.allclose(c.float(), ref.float(), rtol=2e-2, atol=2e-2)
    pwr, clk = measure_sustained_telemetry(bound.run)

    tflops = (2.0 * (m**3) / (med_ms * 1e-3)) / 1e12
    hip_best = TUNED_HIPBLASLT_TFLOPS.get(m, 0.0)
    return {
        "size": m,
        "name": kern.name,
        "cfg": cfg,
        "grid": grid[0],
        "vgpr": kern.manifest.vgpr,
        "agpr": kern.manifest.agpr,
        "sgpr": kern.manifest.sgpr,
        "lds": kern.manifest.lds,
        "resident": kern.manifest.resident,
        "max_abs_err": round(max_abs, 6),
        "mean_abs_err": round(mean_abs, 6),
        "median_ms": round(med_ms, 5),
        "p10_ms": round(p10_ms, 5),
        "p90_ms": round(p90_ms, 5),
        "tflops": round(tflops, 2),
        "hipblaslt_tflops": hip_best,
        "ratio_vs_hipblaslt": round(tflops / hip_best, 4) if hip_best else None,
        "pct_measured_roofline": round(100.0 * tflops / MEASURED_ROOFLINE_TFLOPS, 2),
        "pct_datasheet_peak": round(100.0 * tflops / DATASHEET_PEAK_TFLOPS, 2),
        "power_w": pwr,
        "gfx_mhz": clk,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    with targeting("gfx942"):
        before = thin.use_store(None)
        try:
            records = []
            for m in args.sizes:
                torch.manual_seed(942 + m)
                a = (torch.randn((m, m), device="cuda", dtype=torch.float32) * 0.125).to(
                    torch.bfloat16
                )
                b = (torch.randn((m, m), device="cuda", dtype=torch.float32) * 0.125).to(
                    torch.bfloat16
                )
                ref = torch.matmul(a, b)
                c = torch.empty((m, m), device="cuda", dtype=torch.bfloat16)

                rec = benchmark_config(
                    m, default_gemm_config(m), a, b, c, ref, warmup=args.warmup, iters=args.iters
                )
                records.append(rec)
                print(
                    f"M={m:5d} | thin={rec['tflops']:6.1f} TFLOPS ({rec['median_ms']:.4f} ms) | "
                    f"hipBLASLt={rec['hipblaslt_tflops']:5.1f} | "
                    f"ratio={rec['ratio_vs_hipblaslt']:.3f}x | "
                    f"roof={rec['pct_measured_roofline']:5.1f}% | "
                    f"peak={rec['pct_datasheet_peak']:5.1f}% | "
                    f"pwr={rec['power_w']}W clk={rec['gfx_mhz']}MHz",
                    flush=True,
                )
                del a, b, c, ref
                torch.cuda.empty_cache()

            if args.out:
                with open(args.out, "w") as f:
                    json.dump(records, f, indent=2)
        finally:
            thin.use_store(before)


if __name__ == "__main__":
    main()
