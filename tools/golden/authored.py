"""The source/program generation scope, separate from full-data ownership.

The full --coverage check intentionally includes recorded numeric data. This
inventory identifies the narrower authored-program task without exempting any
file from that check. Generated footer text, measured numeric results, target
facts and encoding/format tables are not hand-written input programs.
"""

from __future__ import annotations

from tools.golden import ROOT, data_files, decode


def _crate(component, names):
    owner = {"target": "targets"}.get(component, component)
    return {f"crates/thin-{owner}/tests/data/{component}/{name}.json" for name in names}


AUTHORED_FILES = (
    _crate("dsl", ("builder", "entry", "expr", "flow", "lanes", "lds", "maker", "values", "waits"))
    | _crate(
        "codegen", ("abi", "kernels", "manifest", "metadata", "printer", "refusals", "verifier")
    )
    | _crate(
        "ir",
        (
            "cfg",
            "exec",
            "flight",
            "instr",
            "loc",
            "model",
            "passes",
            "m0",
            "solvers",
            "stream",
            "verify",
            "waves",
        ),
    )
    | _crate("passes", ("alias", "checks", "hazards", "lives", "ranges", "regalloc", "waits"))
    | _crate("target", ("hazards", "line_syntax", "read_s"))
    | _crate("emu", ("claims", "reading"))
    | {f"tests/data/opspec/{name}.json" for name in ("group", "implement", "runner", "spec")}
    | {f"tests/data/runtime/{name}.json" for name in ("device", "kernel")}
    | {f"tests/data/lib/{name}.json" for name in ("gfx1201", "traces")}
)

# New source-bearing files must not silently escape this explicit inventory.
# Other program notations (IR lines, claim text, etc.) are explicitly listed
# above: a generic "text" key would confuse factual printed/encoded outputs
# with hand-written inputs.
SOURCE_KEYS = {"src", "setup", "steps_setup", "guards_setup", "other_setup"}


def has_source(value):
    """Recognize unambiguous executable-source fields in any future fixture."""
    if isinstance(value, dict):
        return any(
            (key in SOURCE_KEYS and isinstance(item, str) and item.strip()) or has_source(item)
            for key, item in value.items()
        )
    return isinstance(value, list) and any(has_source(item) for item in value)


def coverage(owners, *, root=ROOT):
    """Return authored-only gaps without changing the full ownership check."""
    present = data_files(root)
    discovered = {
        path
        for path in present
        if path.endswith((".json", ".json.gz"))
        and path not in AUTHORED_FILES
        and has_source(decode((root / path).read_bytes(), path))
    }
    expected = AUTHORED_FILES | discovered
    return {
        "unowned": sorted(expected - set(owners)),
        "missing": sorted(expected - present),
        "unclassified_source": sorted(discovered),
    }


# These mixed files also carry numeric-only tables. They are outside the
# program generation, just like the entirely factual files. Only the CLI's
# post-generation comparison/installation layer may preserve their records.
RECORDED_SECTIONS = {
    "crates/thin-codegen/tests/data/codegen/abi.json": ("residency", "budget"),
    "crates/thin-targets/tests/data/target/hazards.json": ("delay_text",),
}
