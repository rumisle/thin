# Performance

## Measuring on an RX 9070 XT

The RX 9070 XT has 16 GB of device memory, 32 WGPs and 64 CUs. Use a
release build for compiler throughput measurements; editable installs use
an incremental profile optimized for rebuild latency instead.

```sh
uv run --with maturin maturin develop --release
uv run --no-sync pytest -m gfx1201 tests/hw/test_run.py
```

For 2^26 f32 elements (768 MiB read and written per call), an RX 9070 XT
measures **1313.1 µs** with `Bound.time()` and **1313.0 µs** with a HIP
event pair, each over 20 repetitions. The command above runs this
comparison. Timing depends on clocks, power limits and concurrent
workloads; report input size, warmup, repetition count and the installed
ROCm/torch versions with any result.
Use `Bound.time()` for repeated device execution, and metron when candidates
need randomized comparisons, input restoration and correctness checks.

Test time and memory ceilings are in `tests/budgets.toml`; they are
regression guards, not throughput claims. A slower machine scales the time
ones explicitly ([ci.md](ci.md#test-time-budgets-on-a-slower-machine)).
Build wheels with the `dist` profile for size and runtime benchmarking.

## Emulator throughput

The ignored `throughput` test in `crates/thin-emu/tests/emu_kernels.rs`
runs vadd (256-lane workgroups) over 2^18 elements on one emulated machine and prints
instructions and lane operations per second:

```sh
cargo test -p thin-emu --release --test emu_kernels throughput -- --ignored --nocapture
```

On a Ryzen 9 9950X it runs 131k instructions in about 0.09 s: 1.4 M
instructions/s, 46 M lane-ops/s. `THIN_EMU_LOG2N=22` sets a larger size
(2^22 elements: 1.2 M instructions/s).

## Register assignment and stream checks on large kernels

`lift_bench` (see [lift.md](lift.md)) times each pass once per kernel on
user-supplied code objects. On a Ryzen 9 9950X, for a large GEMM code
object from a vendor library (23k instructions; the placed program has
about 30k rows and 337 call contexts):

| pass | time |
|---|---|
| register assignment | 42 ms |
| placement check | 42 ms |
| dead-write findings | 10 ms |

Of each assignment, the context- and lane-aware liveness
(`thin_ir::live`) takes about 9 ms; most of the rest is collecting the
per-instruction facts and the loads in flight. The per-kernel digest
`lift_bench` prints lets a change that should keep placements, check
verdicts and dead writes the same be compared before and after.
