# Tutorials

A way into thin for a reader who knows Python and roughly what a GPU is,
but has never written assembly. Each part builds on the one before; each
runs as written, top to bottom, as one script, and every output shown is
what it printed (`tests/test_docs.py` runs them). The blocks marked as
GPU need a gfx1201 GPU and PyTorch; everything else runs on the CPU, on
thin's emulator.

| Part | Teaches |
|---|---|
| [1. Vector add](01-vector-add.md) | Waves, lanes and registers; a kernel, compiled, emulated and run on the GPU; the listing read line by line: values, the lane region, loads and their wait. |
| [2. What the compiler does](02-what-the-compiler-does.md) | Placement, wait counts and hazard padding; what it never does (reorder, optimize, guess); an edit changing exactly the lines it touches. |
| [3. Loops and reductions](03-loops-and-reductions.md) | A row sum: counted loops, waits inside loops and across their back edge, reductions across lanes, loads kept in flight. |
| [4. Shared memory](04-shared-memory.md) | The LDS, its layouts and barriers; a transpose; races refused by the compiler and caught by the emulator; bank conflicts, measured. |
| [5. Matrix cores](05-matrix-cores.md) | One WMMA tile: fragments and their layout, checked against numpy and on the GPU. |
| [6. A tiled matrix multiply](06-tiled-matmul.md) | Global memory to LDS to WMMA, padding, double buffering: four steps, each measured against torch. |
| [7. Measuring](07-measuring.md) | What `Bound.time` measures, how much it moves, comparing variants, and a kernel's own clock (laps). |

The language reference, [../lang.md](../lang.md), says everything the
language does, with the rules behind each refusal the tutorials meet.
