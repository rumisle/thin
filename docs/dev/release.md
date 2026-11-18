# Releasing

thin is published to PyPI as one abi3 wheel per platform (Linux x86_64,
macOS arm64), for CPython 3.10 and later, and an sdist. The Rust crates are
not published (`publish = false`).

## Versions

The version has one source, `[workspace.package] version` in `Cargo.toml`:
the Python package reads it at build time (`dynamic = ["version"]`), and
`thin.__version__` reads it from the installed metadata
(`tests/test_version.py` holds them equal).

Versions follow semantic versioning. Before 1.0, a release that breaks
the API raises the minor version (0.1 to 0.2) and any other raises the
patch version.

## Publishing a release

1. Set the version in `Cargo.toml` (`cargo check` updates `Cargo.lock`) and
   move the `Unreleased` notes of `CHANGELOG.md` under it.
2. Commit as `release: v<version>` and push to `main`; wait for CI.
3. Tag and push: `git tag v<version> && git push origin v<version>`.

The tag runs `.github/workflows/release.yml`: it checks the tag against
`Cargo.toml`, builds the wheels and the sdist, runs the suite against each
wheel, and uploads them to PyPI from the `pypi` environment once it is
approved.

Running the workflow by hand (Actions, release, Run workflow) does the
same and uploads to TestPyPI instead, from the `testpypi` environment: a
rehearsal before the first release, or after changing the workflow.
TestPyPI, like PyPI, takes each version once.

## One-time setup

- On PyPI and on TestPyPI, add a trusted publisher to the `thin` project:
  owner `rumisle`, repository `thin`, workflow `release.yml`, environment
  `pypi` (TestPyPI: `testpypi`). No token is stored anywhere.
- On GitHub, create the environments `pypi` and `testpypi`; give `pypi`
  a required reviewer, so an upload waits for approval.
