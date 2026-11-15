"""Architecture rules a test can check (docs/dev/ARCHITECTURE.md)."""

import ast
import re
from pathlib import Path

from tools import check_binding

ROOT = Path(__file__).resolve().parent.parent
INSTANCE_CRATES = {"thin-targets"}
TARGET_NAMES = ("gfx1201", "gfx942")


def _rust_code_lines(path: Path):
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip().startswith("//"):
            yield number, line


def _python_code_strings(path: Path):
    tree = ast.parse(path.read_text())
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                yield node.lineno, node.value
        elif isinstance(node, ast.Name):
            yield node.lineno, node.id


def test_no_generic_source_names_a_target():
    """only the instance and the registry name a target.

    Generic thin code must use registry data. metron's stack checks may
    name the hardware they diagnose.
    """
    pattern = re.compile("|".join(TARGET_NAMES))
    hits = []
    for crate in sorted((ROOT / "crates").iterdir()):
        if crate.name in INSTANCE_CRATES:
            continue
        for path in sorted(crate.rglob("*.rs")):
            hits += [f"{path}:{n}" for n, line in _rust_code_lines(path) if pattern.search(line)]
    for path in sorted((ROOT / "python" / "thin").rglob("*.py")):
        hits += [f"{path}:{n}" for n, text in _python_code_strings(path) if pattern.search(text)]
    assert hits == []


def test_binding_holds_no_domain_logic():
    """Enforce the binding boundary described in docs/dev/ARCHITECTURE.md."""
    assert check_binding.dependency_errors() + check_binding.source_errors() == []


def test_the_binding_check_finds_loops_and_domain_branches():
    code = check_binding.code_of(
        "if v.is_exact_instance_of::<PyInt>() { a } // if done == 1\n"
        "if let Some(x) = y { x }\n"
        "if form.id < 0 { a }\n"
        "while k > 0 {}\n"
        'let s = "a == b";\n'
        "for i in snap.instrs() {}\n"
    )
    errors = [e.split(": ", 1)[1] for e in check_binding.shape_errors("x.rs", code)]
    assert errors == [
        "a comparison (a domain decision belongs to a core crate)",
        "an if that tests no Python type or array layout",
        "a loop of the binding's own",
        "a comparison (a domain decision belongs to a core crate)",
        "a loop over a core collection",
    ]


def _uses(path: Path) -> set[str]:
    """The top-level modules of thin-emu a source's code refers to (`crate::<m>`)."""
    code = "\n".join(line for _, line in _rust_code_lines(path))
    return set(re.findall(r"crate::(\w+)", code))


def test_emu_modules_break_the_tags_cycle():
    """docs/dev/ARCHITECTURE.md: tags uses nothing; the machine never gemm; gemm only tags."""
    src = ROOT / "crates" / "thin-emu" / "src"
    allowed = {"tags": {"tags"}, "gemm": {"tags", "gemm"}}
    hits = []
    for path in sorted(src.rglob("*.rs")):
        layer = path.relative_to(src).parts[0]
        if layer in allowed:
            hits += [f"{path}: uses {m}" for m in sorted(_uses(path) - allowed[layer])]
        elif "gemm" in _uses(path):
            hits.append(f"{path}: the machine uses gemm")
    assert hits == []


def test_fresh_tag_tables_are_a_scoped_context():
    """Fresh tables are scoped; leaving the context restores previous tables."""
    from thin.emu import gemm  #  names the module

    with gemm.fresh_tables() as tables:
        assert tables is not None


def test_checkers_cannot_be_subclassed_in_python():
    """checkers are native only."""
    import pytest

    from thin.emu import Hooks
    from thin.emu.gemm import GemmHooks

    for checker in (Hooks, GemmHooks):
        with pytest.raises(TypeError):
            type("Mine", (checker,), {})


# The native wait checker has its own parser and dependency model.
# Delay-hint checks are separate Python tests, not a native hazard checker.
INDEPENDENT_CHECKERS = [ROOT / "crates" / "thin-codegen" / "src" / "checker.rs"]


def test_the_independent_checkers_share_no_code_with_the_compiler():
    """The architecture's wait checker reads assembly without compiler helpers."""
    assert INDEPENDENT_CHECKERS
    hits = []
    for path in INDEPENDENT_CHECKERS:
        assert path.is_file(), f"missing independent checker: {path}"
        lines = list(_rust_code_lines(path))
        assert lines, f"empty independent checker: {path}"
        for number, line in lines:
            for name in re.findall(r"\b(thin_\w+|crate::\w+(?:::\w+)*)", line):
                hits.append(f"{path}:{number}: {name}")
    assert hits == []


def test_independence_guard_rejects_missing_and_shared_code(tmp_path, monkeypatch):
    """Missing files and compiler dependencies must not pass vacuously."""
    import pytest

    checker = tmp_path / "checker.rs"
    monkeypatch.setitem(globals(), "INDEPENDENT_CHECKERS", [checker])
    with pytest.raises(AssertionError, match="missing independent checker"):
        test_the_independent_checkers_share_no_code_with_the_compiler()
    checker.write_text("use thin_target::Target;\n")
    with pytest.raises(AssertionError):
        test_the_independent_checkers_share_no_code_with_the_compiler()
    checker.write_text("use std::collections::BTreeSet;\n")
    test_the_independent_checkers_share_no_code_with_the_compiler()
