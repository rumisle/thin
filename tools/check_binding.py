"""Checks mechanically that the binding crate holds no domain logic.

The binding (crates/thin-py) may convert types, map errors and manage object
lifetime, nothing else. Its size grows with the API it binds, so no total
is capped; what a script can check is that each piece stays small and
shaped like conversion:

1. Dependencies: only PyO3, rust-numpy (zero-copy arrays) and thin's own
   crates.
2. Layout: one module per component (`<component>.rs`, or a directory
   `<component>/` of files), so each component's binding is found in one
   place and reviewed in one place. lib.rs holds the shared helpers and
   the module's init, and binds nothing itself.
3. Size: every file within a line budget and every function within a
   length limit, so logic has nowhere to hide.
4. No loops of the binding's own: no `while` or `loop`, and no `for` over a
   core collection (`.instrs`, `.forms`, `.values`, `.layout()`, `.rows`).
   Converting a list is an iterator adapter over the host value.
5. No domain branching: no comparison operators, and an `if` (or a match
   guard) may test only a Python value's type or an array's layout. A
   decision about the domain belongs to a core crate; `if let` and `match`
   destructure, which conversion needs.

Review covers the rest. Run: python -m tools.check_binding
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CRATE = ROOT / "crates" / "thin-py"
ALLOWED_DEPENDENCIES = re.compile(r"^(pyo3|numpy|thin-[a-z0-9-]+)$")
COMPONENTS = frozenset(
    {
        "formats",
        "target",
        "ir",
        "passes",
        "dsl",
        "codegen",
        "emu",
        "emu_tags",
        "runtime",
        "opspec",
        "library",
    }
)
"""The components a binding module may be named after (the kernel
library is `library`, since `lib.rs` is the crate root)."""
FILE_LINE_LIMIT = 400
FUNCTION_LINE_LIMIT = 40
CORE_COLLECTION_LOOP = re.compile(r"\bfor\b[^\n]*\.(instrs|forms|values|layout\(\)|rows)\b")
OWN_LOOP = re.compile(r"\b(while|loop)\b")
COMPARISON = re.compile(r"==|!=|<=|>=|\s<\s|\s>\s")
CONDITION = re.compile(r"\bif\b(?!\s+let\b)([^{\n]*)")
TYPE_TEST = re.compile(
    r"\b(is_none|is_instance_of|is_exact_instance_of|is_c_contiguous|contiguous)\b"
)
BINDS = re.compile(r"#\[(pyfunction|pyclass|pymethods)")


def dependency_errors() -> list[str]:
    """Dependencies of the binding other than PyO3, rust-numpy and thin crates."""
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    package = next(p for p in json.loads(out.stdout)["packages"] if p["name"] == "thin-py")
    return [
        f"thin-py depends on {d['name']}: only pyo3, numpy and thin crates are allowed"
        for d in package["dependencies"]
        if d["kind"] is None and not ALLOWED_DEPENDENCIES.match(d["name"])
    ]


def code_of(text: str) -> str:
    """The source with comments and string literals blanked, lines kept."""
    text = re.sub(r'"(\\.|[^"\\])*"', '""', text)
    return re.sub(r"//[^\n]*", "", text)


def function_lengths(text: str) -> list[tuple[str, int]]:
    """(name, lines) of each `fn` in a Rust source, by brace matching."""
    lengths = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        match = re.match(r"\s*(pub(\([a-z]+\))? )?fn (\w+)", line)
        if not match:
            continue
        depth, j = 0, i
        while j < len(lines):
            depth += lines[j].count("{") - lines[j].count("}")
            if depth == 0 and "{" in "".join(lines[i : j + 1]):
                break
            j += 1
        lengths.append((match.group(3), j - i + 1))
    return lengths


def layout_errors(paths: list[Path]) -> list[str]:
    """Files that are not lib.rs or a component's module."""
    errors = []
    for path in paths:
        relative = path.relative_to(CRATE / "src")
        component = relative.parts[0].removesuffix(".rs")
        if relative == Path("lib.rs"):
            if BINDS.search(code_of(path.read_text())):
                errors.append(
                    "lib.rs: binds a function or class; it belongs to a component's module"
                )
        elif component not in COMPONENTS or len(relative.parts) > 2:
            errors.append(f"{relative}: not a component's module ({', '.join(sorted(COMPONENTS))})")
    return errors


def shape_errors(name: str, code: str) -> list[str]:
    """Loops and branches of the binding's own in one file's code."""
    errors = []
    for number, line in enumerate(code.splitlines(), 1):
        where = f"{name}:{number}"
        if CORE_COLLECTION_LOOP.search(line):
            errors.append(f"{where}: a loop over a core collection")
        if OWN_LOOP.search(line):
            errors.append(f"{where}: a loop of the binding's own")
        if COMPARISON.search(line):
            errors.append(f"{where}: a comparison (a domain decision belongs to a core crate)")
        condition = CONDITION.search(line)
        if condition and not TYPE_TEST.search(condition.group(1)):
            errors.append(f"{where}: an if that tests no Python type or array layout")
    return errors


def source_errors() -> list[str]:
    """Layout, size and shape violations in the binding's sources."""
    paths = sorted((CRATE / "src").rglob("*.rs"))
    errors = layout_errors(paths)
    for path in paths:
        name = str(path.relative_to(CRATE / "src"))
        text = path.read_text()
        lines = len(text.splitlines())
        if lines > FILE_LINE_LIMIT:
            errors.append(
                f"{name} has {lines} lines (> {FILE_LINE_LIMIT}): split the component's module"
            )
        for fn, length in function_lengths(text):
            if length > FUNCTION_LINE_LIMIT:
                errors.append(f"{name}: fn {fn} has {length} lines (> {FUNCTION_LINE_LIMIT})")
        errors.extend(shape_errors(name, code_of(text)))
    return errors


def main() -> int:
    """Prints every violation; non-zero when there is one."""
    errors = dependency_errors() + source_errors()
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
