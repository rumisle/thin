"""The documentation's examples, run and compared with the output under each.

Every ```python block of docs/lang.md and docs/tutorials/*.md runs. The
output block right after it (```asm, ```error or ```text, with only blank
lines between) is what it must produce; a block without one must produce
nothing.

- docs/lang.md: each block runs alone, on a fresh gfx1201 Builder ``b``,
  with ``V`` and ``S`` its files and every name of thin.dsl imported.
  ```asm is the rendered kernel (``codegen.render``) after what the block
  printed, ```text what it printed, ```error the refusal it raised
  (``Class: message``) while it ran or when the kernel was rendered.
- docs/tutorials/*.md: a file's blocks run in order in one namespace, as
  the parts of one script. ```asm and ```text are what a block printed and
  the warnings it raised (``Category: message``), ```error the exception
  it ended with.

Each block is a file ``kernel.py`` of its own: a location in a message is
a line of its block. A ```python gpu block needs the GPU: it is marked
gfx1201, and the numbers of its output (timings, counts of cycles) match
any number. Compiles go through no store.

``THIN_DOCS_UPDATE=1`` writes each block's actual output in place of the
one under it: outputs are produced, never typed.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import tempfile
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
OUTPUTS = ("asm", "error", "text")
UPDATE = os.environ.get("THIN_DOCS_UPDATE") == "1"
NUMBER = re.compile(r"\d+(?:\.\d+)?(?:e[-+]?\d+)?")
FRAME = re.compile(r"^    at (\S+):(\d+ \(.*\))$")


@dataclass
class Block:
    """A python block and the output block under it."""

    path: Path
    line: int
    """The python block's first code line (1-based)."""
    code: str
    gpu: bool
    kind: str | None
    """The output block's language, or None."""
    want: str | None
    span: tuple[int, int] | None
    """The output block's code lines [start, end) (0-based)."""

    @property
    def id(self) -> str:
        return f"{self.path.relative_to(DOCS)}:{self.line}"

    @property
    def fresh(self) -> bool:
        return self.path.name == "lang.md"


def _fences(lines):
    """(info, first code line, end line) of every fenced block (0-based)."""
    k = 0
    while k < len(lines):
        m = re.match(r"^```(\S.*)?$", lines[k])
        if m is None:
            k += 1
            continue
        info, start = (m.group(1) or "").strip(), k + 1
        end = start
        while end < len(lines) and lines[end] != "```":
            end += 1
        if end == len(lines):
            raise ValueError(f"line {k + 1}: an unclosed block")
        yield info, start, end
        k = end + 1


def blocks(path: Path) -> list[Block]:
    """The python blocks of a markdown file, each with its output block."""
    lines = path.read_text().split("\n")
    fences = list(_fences(lines))
    out = []
    for j, (info, start, end) in enumerate(fences):
        words = info.split()
        if not words or words[0] != "python":
            continue
        if any(w != "gpu" for w in words[1:]):
            raise ValueError(f"{path.name}:{start}: ```{info}: python or python gpu")
        kind = want = span = None
        if j + 1 < len(fences):
            info2, s2, e2 = fences[j + 1]
            between = lines[end + 1 : s2 - 1]
            if info2 in OUTPUTS and all(not x.strip() for x in between):
                kind, want, span = info2, "\n".join(lines[s2:e2]), (s2, e2)
        code = "\n".join(lines[start:end]) + "\n"
        out.append(Block(path, start + 1, code, "gpu" in words[1:], kind, want, span))
    return out


def files() -> list[Path]:
    """The documents whose examples run."""
    return [DOCS / "lang.md", *sorted((DOCS / "tutorials").glob("*.md"))]


ALL = [b for f in files() if f.exists() for b in blocks(f)]


# --- Running ----------------------------------------------------------------

_TMP = tempfile.TemporaryDirectory(prefix="thin-docs-")


def _file(block: Block) -> str:
    """The block's own kernel.py, kept for the session (a generator's
    identity is its file's bytes)."""
    d = Path(_TMP.name) / f"{block.path.stem}-{block.line}"
    d.mkdir(exist_ok=True)
    p = d / "kernel.py"
    p.write_text(block.code)
    return str(p)


def _message(e: BaseException) -> str:
    """``Class: message``, without the frames of this harness a refusal's
    source chain names (a script's own module frame is its outermost)."""
    lines = []
    for x in str(e).split("\n"):
        m = FRAME.match(x)
        if m is None:
            lines.append(x)
        elif os.path.basename(m.group(1)) == "kernel.py":
            lines.append(f"    at kernel.py:{m.group(2)}")
    return f"{type(e).__name__}: " + "\n".join(lines)


def _run(block: Block, scope: dict) -> str:
    """The block's output: what it printed, then its render, warning or error."""
    from thin import codegen

    code = compile(block.code, _file(block), "exec")
    printed = io.StringIO()
    tail: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with contextlib.redirect_stdout(printed):
                exec(code, scope)
                if block.fresh and block.kind in ("asm", "error"):
                    tail = list(codegen.render(scope["b"]))
        except Exception as e:  # the output an ```error block shows
            if block.kind != "error":
                raise AssertionError(
                    f"{block.id}: raised\n{''.join(traceback.format_exception(e))}"
                ) from None
            tail = [_message(e)]
    shown = [f"{w.category.__name__}: {w.message}" for w in caught]
    text = printed.getvalue()
    if not block.fresh and shown:
        text += "\n".join(shown) + "\n"
    return (text + "\n".join(tail)).rstrip()


def _fresh_scope() -> dict:
    from thin import dsl

    b = dsl.Builder()
    scope = {name: getattr(dsl, name) for name in dsl.__all__}
    scope.update(b=b, V=b.v, S=b.s, __name__="kernel")
    return scope


_SESSIONS: dict[Path, tuple[dict, int]] = {}
"""Per tutorial: its namespace and the index of the last block run in it."""


def _gpu() -> bool:
    from tests.conftest import _gfx1201

    return _gfx1201() is not None


def _session(block: Block) -> dict:
    """The tutorial's namespace with every block before this one run."""
    mine = [b for b in ALL if b.path == block.path]
    k = mine.index(block)
    scope, last = _SESSIONS.get(block.path, (None, -1))
    if scope is None or last >= k:
        scope, last = {"__name__": "kernel"}, -1
    for before in mine[last + 1 : k]:
        if not before.gpu or _gpu():
            _run(before, scope)
    _SESSIONS[block.path] = (scope, k)
    return scope


def _same(block: Block, got: str) -> bool:
    want = block.want or ""
    if not block.gpu:
        return got == want.rstrip()
    pattern = "".join(
        NUMBER.pattern if NUMBER.fullmatch(part) else re.escape(part)
        for part in re.split(f"({NUMBER.pattern})", want.rstrip())
    )
    return re.fullmatch(pattern, got) is not None


def _write(block: Block, got: str) -> None:
    """Puts got in place of the block's output (the file read again: earlier
    updates moved its lines; the block is the same by its place)."""
    k = [b for b in ALL if b.path == block.path].index(block)
    now = blocks(block.path)[k]
    if now.code != block.code or now.span is None:
        raise AssertionError(f"{block.id}: the file changed under the update")
    lines = block.path.read_text().split("\n")
    s, e = now.span
    lines[s:e] = got.split("\n") if got else []
    block.path.write_text("\n".join(lines))


@pytest.fixture
def _no_store():
    import thin
    from thin.target import targeting

    before = thin.use_store(None)
    with targeting("gfx1201"):
        yield
    thin.use_store(before)


def _param(b: Block):
    marks = [pytest.mark.gfx1201] if b.gpu else []
    return pytest.param(b, id=b.id, marks=marks)


@pytest.mark.parametrize("block", [_param(b) for b in ALL])
def test_example_output(_no_store, block):
    scope = _fresh_scope() if block.fresh else _session(block)
    got = _run(block, scope)
    if block.kind is None:
        if block.fresh:
            pytest.fail(f"{block.id}: an example of the reference shows its output")
        assert got == "", f"{block.id}: printed, and has no output block:\n{got}"
        return
    if UPDATE and got != (block.want or "").rstrip():
        _write(block, got)
        return
    if _same(block, got):
        return
    pytest.fail(f"{block.id}: produced\n{got}\n\nthe document shows\n{block.want}")


# --- The reference's coverage ---------------------------------------------------

INTERNAL = {
    "Tree": "the type of an expression; the reference describes expressions",
}
"""Exported names the reference leaves out on purpose, and why."""


def test_the_reference_names_every_export():
    from thin import dsl

    if not (DOCS / "lang.md").exists():
        pytest.skip("no docs/lang.md yet")
    text = re.sub(r"(?ms)^```.*?^```$", "", (DOCS / "lang.md").read_text())
    spans = re.findall(r"`([^`\n]+)`", text)
    missing = [
        n
        for n in dsl.__all__
        if n not in INTERNAL and not any(re.search(rf"\b{n}\b", x) for x in spans)
    ]
    assert not missing, f"docs/lang.md does not document: {', '.join(missing)}"


def test_a_block_is_read_with_the_output_under_it(tmp_path):
    doc = tmp_path / "t.md"
    doc.write_text(
        "# t\n\n```python\nprint(1)\n```\n\n```text\n1\n```\n\n"
        "```python gpu\nx = 2\n```\n\ntext\n\n```text\n2\n```\n"
    )
    first, second = blocks(doc)
    assert (first.line, first.code, first.gpu, first.kind, first.want) == (
        4,
        "print(1)\n",
        False,
        "text",
        "1",
    )
    assert (second.line, second.gpu, second.kind) == (12, True, None)


def _words(line: str) -> str:
    return " ".join(line.split())


def test_every_quote_is_of_an_output_above_it():
    """A ```asm quote block repeats lines of an output block above it in its
    document, in order (spaces aside): a listing read a few lines at a time."""
    for path in files():
        if not path.exists():
            continue
        lines = path.read_text().split("\n")
        shown: list[list[str]] = []
        for info, start, end in _fences(lines):
            body = [_words(x) for x in lines[start:end]]
            if info in OUTPUTS:
                shown.append(body)
            elif info == "asm quote":
                n = len(body)
                found = any(out[k : k + n] == body for out in shown for k in range(len(out)))
                assert found, f"{path.name}:{start}: quotes no output above it"


def test_every_output_block_follows_an_example():
    for path in files():
        if not path.exists():
            continue
        lines = path.read_text().split("\n")
        fences = list(_fences(lines))
        for j, (info, start, _) in enumerate(fences):
            if info in OUTPUTS:
                before = fences[j - 1][0].split()[:1] if j else []
                assert before == ["python"], f"{path.name}:{start}: ```{info} under no example"


# --- The operator table -----------------------------------------------------------

COLUMNS = (("V", "u32"), ("V", "i32"), ("V", "f32"), ("S", "u32"), ("S", "i32"))
COMPARE = re.compile(r"^~?\(?a [<>=!]=? c\)?$")


def _table_rows():
    """(expression, [cell per column]) of the operator table of docs/lang.md."""
    text = (DOCS / "lang.md").read_text()
    head = text.index("| Expression | `V` u32 |")
    for line in text[head:].split("\n")[2:]:
        if not line.startswith("|"):
            return
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line)[1:-1]]
        yield cells[0].strip("`").replace("\\|", "|"), cells[1:]


def _mnemonics(expr: str, file: str, ty: str):
    """The instructions the expression renders, hazards aside, or None if refused."""
    from thin import codegen, dsl
    from thin.ir import Diag

    b = dsl.Builder()
    f = b.files[file.lower()]
    scope = {name: getattr(dsl, name) for name in dsl.__all__}
    scope.update(a=getattr(f, ty)(name="a", at=2), c=getattr(f, ty)(name="c", at=3))
    try:
        tree = eval(expr, scope)
        if COMPARE.match(expr):
            b.s.mask(tree, name="r")
        else:
            getattr(f, ty)(tree, name="r")
        lines = codegen.render(b)
    except Diag:
        return None
    hazard = ("s_delay_alu", "s_wait_alu", "v_nop", "s_nop")
    return [x.split()[0] for x in lines if not x.split()[0].startswith(hazard)]


@pytest.mark.skipif(not (DOCS / "lang.md").exists(), reason="no docs/lang.md yet")
def test_the_operator_table_is_what_renders(_no_store):
    rows = list(_table_rows())
    assert rows
    for expr, cells in rows:
        for (file, ty), cell in zip(COLUMNS, cells, strict=True):
            want = None if cell == "refused" else re.findall(r"`([a-z0-9_]+)`", cell)
            got = _mnemonics(expr, file, ty)
            assert got == want, f"{expr} ({file} {ty}): the table says {cell}, it renders {got}"


# --- Links ------------------------------------------------------------------------


def _anchor(heading: str) -> str:
    """GitHub's anchor of a heading."""
    a = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return a.replace(" ", "-")


def test_every_link_of_the_docs_resolves():
    for path in [*DOCS.rglob("*.md"), ROOT / "README.md"]:
        text = re.sub(r"(?ms)^```.*?^```$", "", path.read_text())
        anchors = {_anchor(h) for h in re.findall(r"(?m)^#+ (.+)$", text)}
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if re.match(r"[a-z]+://", target):
                continue
            file, _, frag = target.partition("#")
            where = (path.parent / file).resolve() if file else path
            assert where.exists(), f"{path.relative_to(ROOT)}: {target}: no such file"
            if frag and where.suffix == ".md":
                there = re.sub(r"(?ms)^```.*?^```$", "", where.read_text())
                names = {_anchor(h) for h in re.findall(r"(?m)^#+ (.+)$", there)}
                assert frag in (names if file else anchors), (
                    f"{path.relative_to(ROOT)}: {target}: no such heading"
                )
