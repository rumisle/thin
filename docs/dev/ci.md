# Continuous integration

`.github/workflows/ci.yml` runs plain commands a developer can run
locally. Two things in it are decisions rather than commands: which Rust
builds thin, and how the test-time budgets apply to a slower machine.

## The Rust toolchain

`rust-toolchain.toml` pins an exact release (`channel = "1.100.0"`), not
`stable`. Clippy gains lints with every release, and CI runs it with
`-D warnings`: on a floating channel a new stable fails CI on code nobody
touched. With the pin, rustup (CI's, and a developer's) builds with the
same compiler and clippy, and new lints arrive when someone bumps it.

A distribution's Rust (no rustup) ignores the file. Run the clippy gate
with the pinned toolchain anyway, or its verdict is not CI's; rustup can
live in a private home beside the system Rust:

```sh
curl -sSfLO https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init
chmod +x rustup-init
export RUSTUP_HOME=~/.cache/rustup CARGO_HOME=~/.cache/cargo PATH=~/.cache/cargo/bin:$PATH
./rustup-init -y --no-modify-path --profile minimal --default-toolchain none
rustup toolchain install          # in the checkout: the pinned one, with its components
```

To bump the pin, in one commit of its own:

1. Set `channel` to the new release and run `rustup toolchain install`.
2. Run the gates with it: `cargo fmt --all --check`,
   `cargo clippy --workspace --all-targets -- -D warnings` and
   `cargo test --workspace`. Read cargo's own warnings too (such as
   `unused dependency`), which `-D warnings` does not fail on.
3. Fix what the new release reports. A lint is silenced only with an
   `#[allow]` that says why the code is right as it is.
4. Commit as `[ci] Bump the Rust toolchain to <version>`, with the fixes in
   commits before it if they are large.

Bump when a release brings something thin wants, or every few releases to
stay near stable; never as a side effect of another change.

The pin is not the oldest Rust thin builds with. That is `rust-version` in
`Cargo.toml` (the MSRV), and CI's `msrv` job checks the workspace, every
target, with exactly that release and the checked-in `Cargo.lock`. Raising
the MSRV is a decision of its own, made in `Cargo.toml`; the job follows it.

## Test-time budgets on a slower machine

`tests/budgets.toml` bounds the tests' time and memory. Its time budgets
are regression guards calibrated on a fast developer machine (a Ryzen 9
9950X), not claims about every machine. Time is held per test, never for
the whole suite: a test that gets slower is a regression, a suite that
gains tests is not.

A slower machine says how much slower it is in `THIN_BUDGET_SCALE`. Every
time budget (each key ending in `seconds`: a test, a slow section, an
import, a Rust golden binary) is multiplied by it:

- Unset, the budgets are as written. A developer machine never loosens
  them by being slow; the factor has to be set, by hand, where it is meant.
- Set, it must be a number from 1 to 4; anything else fails the run instead
  of being ignored. A machine more than 4x slower is too unlike the
  calibration one for the budgets to guard it.
- It is never silent: pytest's header says `time budgets scaled by
  THIN_BUDGET_SCALE=2`, and every time violation gives the budget as
  scaled and as written (`2 s: 1 s scaled by THIN_BUDGET_SCALE=2`), as
  each Rust golden binary's cost line does.
- Memory budgets are never scaled: a slower machine does not use more.

CI sets `THIN_BUDGET_SCALE: "2"` for the whole workflow. GitHub's
`ubuntu-latest` runners took 9.5-12.3 s for the same suite, about twice the
calibration machine, slower per core rather than for having two cores (the
suite takes as long on two cores of the calibration machine). Measure
again before changing the factor, and keep it the smallest that passes
reliably: each step loosens CI's guard.

When the calibration machine itself changes, change the budgets in
`tests/budgets.toml`, not the factor.
