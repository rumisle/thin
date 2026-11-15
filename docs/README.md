# thin's documentation

- [Tutorials](tutorials/README.md): the way in, from a vector add to a
  tiled matrix multiply on the matrix cores, each part run as written.
- [The language reference](lang.md): everything the language does, what
  compilation derives and checks, and the instructions that come out.

For contributors, in [dev/](dev/):

- [Architecture](dev/ARCHITECTURE.md): the crates, the compilation
  pipeline, encoding and lifting, the emulator's layers, the tests.
- [The emulator's design](dev/emulator.md): its machine, decoded
  programs, numbers, provenance and the GEMM checker.
- [Adding a target](dev/adding-a-target.md): what a new target supplies.
- [Lifting code objects](dev/lift.md): the lift's steps and their laws.
- [Hardware validation](dev/hardware.md): what the GPU suite checks.
- [Performance](dev/perf.md): measurements of the compiler, the emulator
  and a kernel.
- [Limits of the internals](dev/limitations.md).
- [Continuous integration](dev/ci.md): the pinned Rust toolchain and how
  to bump it, the MSRV, and the test-time budgets on a slower machine.
