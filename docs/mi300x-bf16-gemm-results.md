# MI300X (`gfx942`) BF16 GEMM Roofline and Benchmark

Square row-major `C = A @ B` (`bf16` inputs, `f32` accumulation, `bf16` output,
[`examples/gfx942/gemm_bf16.py`](../examples/gfx942/gemm_bf16.py)) measured on
an AMD Instinct MI300X VF (`gfx942:sramecc+:xnack-`, SPX partition, 8 XCDs × 38
CUs = 304 CUs, 1,216 XDL matrix pipes, 4 MiB L2 per XCD, 256 MiB MALL, 750 W
TDP, ROCm 10.1, PyTorch 2.14.0).

## Roofline and method

- **Datasheet peak (`2.10 GHz`)**: `1307.44 TFLOP/s` (`304 * 4 * 512 * 2.10e9`).
- **Measured pure-MFMA ceiling**: back-to-back `v_mfma_f32_16x16x16_bf16` with
  no memory traffic sustains `1132.98 TFLOP/s` at `1866.8 MHz` / `741.2 W`
  (`1140.86 TFLOP/s` short burst).
- **Measured HBM3 streaming copy**: `3854.7 GB/s` (`torch.copy_`) and
  `3605.6 GB/s` (`thin` `buffer_load_dwordx4` / `buffer_store_dwordx4 nt`).
- **Arithmetic intensity**: the ridge point is `1132.98e12 / 3854.7e9 = 293.9 FLOP/B`.
  For `M in {2048, 4096, 8192, 16384}`, `AI = M / 3` is `682.7` to `5461.3 FLOP/B`,
  so all four sizes are compute-bound.
- **Timing protocol**: identical `randn` `bf16` inputs, clocks unpinned, at
  least `30 ms` (and at least `25` calls) of warmup, median of `60` per-call
  HIP event pairs, checked against `torch.matmul` (`rtol=2e-2, atol=2e-2`).
  Socket power and XCD clocks are sampled via `amd-smi` in a separate untimed
  pass so subprocess calls do not perturb the timed launches.
- **Tuned `hipBLASLt` baseline**: offline search over all `1,104`–`1,110`
  supported `gfx942` solutions per size:
  `./hipblaslt_tune --m <M> --n <M> --k <M> --out <bf16|f32> --warmup 25 --iters 60`.
  On this box `bf16` output (`443.6 / 650.2 / 693.9 / 650.1 TFLOP/s`) is faster
  than `f32` output (`442.7 / 625.5 / 667.5 / 632.8 TFLOP/s`) at every size.

## Results

| `M=N=K` | Tuned `hipBLASLt` (`bf16` / `f32`) | `thin` tile (`BMxBNxBK`, `g`) | `thin` TFLOP/s (suite / isolated) | Ratio vs `hipBLASLt` | % measured roofline (`1133.0`) | % datasheet peak (`1307.4`) | Power / XCD clk | Main remaining limiter |
|---|---|---|---|---|---|---|---|---|
| `2048` | `443.6` / `442.7` | `64x256x64`, `g8` (`256` WGs) | `459.0` (`0.0374 ms`) / `464.5` | `1.035x`–`1.047x` | `40.5%`–`41.0%` | `35.1%`–`35.5%` | `730 W` / `1497 MHz` | 304-CU tile quantization (`256/304 = 84.2%` active CUs for `>=8B` coalesced `B` loads) and cold HBM3 fill/drain across 32 `K`-stages |
| `4096` | `650.2` / `625.5` | `112x512x64`, `g6` (`296` WGs) | `664.6` (`0.2068 ms`) / `672.3` | `1.022x`–`1.034x` | `58.7%`–`59.3%` | `50.8%`–`51.4%` | `733 W` / `1281 MHz` | 750 W TDP clock throttling under HBM3/MALL/L2 load (`~1281 MHz` vs `1867 MHz` pure-MFMA) and 1-wave `296/304` CU quantization |
| `8192` | `693.9` / `667.5` | `144x512x64`, `g6` (`912` WGs) | `699.6` (`1.5715 ms`) / `705.1` | `1.008x`–`1.016x` | `61.8%`–`62.2%` | `53.5%`–`53.9%` | `744 W` / `1244 MHz` | 750 W TDP clock throttling (`92.0%` of all 1,216 XDL pipes busy; `0` LDS bank conflicts) |
| `16384` | `650.1` / `632.8` | `160x512x64`, `g12` (`3296` WGs) | `662.3` (`13.2813 ms`) / `665.3` | `1.019x`–`1.023x` | `58.5%`–`58.7%` | `50.7%`–`50.9%` | `750 W` / `1163 MHz` | Hard 750 W TDP cap (`~1155 MHz` effective XCD clock; `93.6%` of all 1,216 XDL pipes busy; `0` LDS bank conflicts) |

*(The suite column runs all four sizes back-to-back with `0.65 s` telemetry passes between sizes; the isolated column measures a single size without prior thermal soak.)*

## `rocprofv3` counters and power-wall analysis

Collected in two non-multiplexed passes (`<= 6` counters per pass) on `gfx942`.
On SPX MI300X, `GRBM_GUI_ACTIVE` sums across the 8 XCD `GRBM` instances (152
SIMDs per XCD), so `SQ_VALU_MFMA_BUSY_CYCLES / (152 * GRBM_GUI_ACTIVE)` gives
chip-wide XDL matrix-pipe utilization from dispatch start to completion:

| `M=N=K` | `SQ_WAVES` | `GRBM_GUI_ACTIVE` (8 XCDs) | Chip-wide XDL busy | `SQ_LDS_BANK_CONFLICT` | L2 hit rate (`TCC`) | `SQ_WAIT_ANY / SQ_BUSY` |
|---|---|---|---|---|---|---|
| `2048` | `1,024` | `527,716` | `41.6%` (`49.4%` on 256 CUs) | `0` | `83.4%` | `99.9%` |
| `4096` | `1,184` | `2,190,760` | `81.6%` (`83.8%` on 296 CUs) | `0` | `78.8%` | `38.9%` |
| `8192` | `3,648` | `15,433,140` | `92.0%` | `0` | `80.1%` | `23.6%` |
| `16384` | `13,184` | `121,783,525` | `93.6%` | `0` | `81.8%` | `14.7%` |

- **750 W socket TDP cap (`4096`, `8192`, `16384`)**: pure MFMA draws no HBM3,
  MALL, or L2 power and sustains `1866.8 MHz`. During GEMM, memory and fabric
  traffic consume roughly `200`–`230 W` of the `750 W` socket budget, lowering
  XCD clocks to `1162`–`1280 MHz`. At `16384` (`~1155 MHz`), clock-adjusted peak
  times measured XDL duty (`93.6%`) and valid-row fraction (`16384 / 16480`) is
  `1307.44 * (1155 / 2100) * 0.936 * (16384 / 16480) = 669 TFLOP/s`, matching
  the measured `662.9`–`665.3 TFLOP/s`.
- **304-CU quantization (`2048`)**: 4-wave `dwordx2` coalesced `B` loads require
  `BN >= 256` (`grid_n = 8` at `N = 2048`). Because `BM` is a multiple of 16,
  `grid_m * 8` steps from `256` WGs (`BM = 64`, `84.2%` of 304 CUs) to `344 > 304`
  WGs (`BM = 48`, spilling into a second wave). Two-wave/SIMD (`wgsize = 512`)
  and `split-K = 2` variants halve VMEM prefetch depth or add workspace traffic
  and measure lower (`301`–`410 TFLOP/s`).

## Reproducing

```sh
# Emulator numerics and zero-bank-conflict checks (no GPU needed):
uv run pytest tests/test_gemm_bf16.py -q

# MI300X GPU correctness, HIP-event benchmark, and telemetry:
python3 -m benchmarks.bench_gemm_bf16 --sizes 2048 4096 8192 16384 --warmup 25 --iters 60
```
