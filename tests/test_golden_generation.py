"""Generation infrastructure must construct outputs before reading fixtures."""

import builtins
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.golden import data_files, decode, encode
from tools.golden.__main__ import changes, main


def test_serialization_is_deterministic():
    doc = {"edge": [0, 1, -1], "unicode": "µ"}
    for compressed in (False, True):
        raw = encode(doc, compressed=compressed)
        assert raw == encode(doc, compressed=compressed)
        assert decode(raw, "file.json.gz" if compressed else "file.json") == doc


def test_changed_paths_name_all_different_fields():
    assert changes({"cases": [{"bits": [1, 2]}]}, {"cases": [{"bits": [1, 3]}, {}]}) == [
        "cases[0].bits[1]",
        "cases[1]",
    ]


def test_coverage_includes_binary_and_python_only_files(tmp_path):
    paths = ["crates/thin-formats/tests/data/one.co", "tests/data/runtime/one.json"]
    for name in paths:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    assert data_files(tmp_path) == set(paths)


def test_coverage_command_fails_for_an_unowned_file(monkeypatch, capsys):
    import tools.golden.__main__ as command

    monkeypatch.setattr(command, "data_files", lambda: {"tests/data/unowned.bin"})
    assert main(["--coverage"]) == 1
    assert "tests/data/unowned.bin" in json.loads(capsys.readouterr().out)["unowned"]


@pytest.mark.parametrize("component", ["target", "lib", "metron", "dsl"])
def test_generation_cannot_open_checked_in_data(monkeypatch, component):
    import importlib

    if component == "metron":
        pytest.importorskip("torch")
    from thin.runtime import kernel

    # Hash afresh: a digest an earlier test made would hide what compiling reads.
    monkeypatch.setattr(kernel, "_SOURCES", {})
    monkeypatch.setattr(kernel, "_DIGESTS", {})
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert "tests/data/" not in path.as_posix(), f"generator read checked-in data: {path}"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    for module in (builtins, io):
        original_open = module.open

        def checked_open(file, *args, _open=original_open, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)):
                path = Path(os.fsdecode(file)).as_posix()
                assert "tests/data/" not in path, f"generator opened checked-in data: {path}"
            return _open(file, *args, **kwargs)

        monkeypatch.setattr(module, "open", checked_open)
    module = importlib.import_module(f"tools.golden.{component}")
    assert set(module.generate()) == set(module.OUTPUTS)


@pytest.mark.parametrize("path", ["a.json", "a.json.gz"])
def test_encoded_results_do_not_depend_on_document_format(path):
    doc = {"negative_zero": "80000000"}
    assert decode(encode(doc, compressed=path.endswith(".gz")), path) == doc


def test_authored_coverage_does_not_weaken_full_coverage(monkeypatch, capsys):
    import tools.golden.__main__ as command
    import tools.golden.authored as authored

    source = "tests/data/new_source.json"
    recorded = "tests/data/recorded.bin"
    monkeypatch.setattr(authored, "AUTHORED_FILES", {source})
    monkeypatch.setattr(authored, "data_files", lambda root: {source, recorded})
    monkeypatch.setattr(command, "data_files", lambda: {source, recorded})
    monkeypatch.setattr(
        command, "generators", lambda: {"example": SimpleNamespace(OUTPUTS=(source,))}
    )
    assert main(["--authored-coverage"]) == 0
    assert json.loads(capsys.readouterr().out)["unowned"] == []
    assert main(["--coverage"]) == 1
    assert json.loads(capsys.readouterr().out)["unowned"] == [recorded]


def test_new_source_file_cannot_escape_authored_inventory(monkeypatch, tmp_path):
    import tools.golden.authored as authored

    source = "tests/data/future.json"
    path = tmp_path / source
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"cases": [{"src": "x = 3"}]}))
    monkeypatch.setattr(authored, "AUTHORED_FILES", set())
    result = authored.coverage({}, root=tmp_path)
    assert result == {"unowned": [source], "missing": [], "unclassified_source": [source]}


def test_measurement_provenance_is_not_program_source():
    from tools.golden.authored import has_source

    assert not has_source({"source": "measured on hardware", "words": [1, 2, 3]})
    assert has_source({"cases": [{"setup": "def kernel(): pass"}]})


@pytest.mark.parametrize("opener", ["builtin", "io", "path", "os"])
def test_construction_guard_refuses_fixture_io(opener, tmp_path):
    from tools.golden import no_checked_data

    path = tmp_path / "tests/data/old.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    # Each opener reaches the file its own way (3.10's Path.open holds its own
    # io.open; os.open bypasses io): the guard must refuse them all.
    with no_checked_data(), pytest.raises(PermissionError, match="test data"):
        if opener == "path":
            path.read_text()
        elif opener == "builtin":
            with builtins.open(path):
                pass
        elif opener == "os":
            os.close(os.open(path, os.O_RDONLY))
        else:
            with io.open(path):  # noqa: UP020 - exercise Path.open's separate backend
                pass
    assert path.read_text() == "{}", "the guard holds only while it is open"


def test_authored_install_preserves_mixed_numeric_tables_after_construction(monkeypatch, tmp_path):
    import tools.golden.__main__ as command

    path = "crates/thin-codegen/tests/data/codegen/abi.json"
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_bytes(encode({"residency": [17], "budget": [23], "args": ["old source"]}))
    constructed = False
    original_read = Path.read_bytes

    def read_after_construction(file):
        assert constructed
        return original_read(file)

    def fresh():
        nonlocal constructed
        constructed = True
        return {path: encode({"residency": [99], "budget": [99], "args": ["new source"]})}

    monkeypatch.setattr(command, "ROOT", tmp_path)
    monkeypatch.setattr(
        command, "generators", lambda: {"codegen": SimpleNamespace(OUTPUTS=(path,), generate=fresh)}
    )
    monkeypatch.setattr(Path, "read_bytes", read_after_construction)
    assert main(["codegen", "--authored"]) == 0
    assert json.loads(target.read_text()) == {
        "residency": [17],
        "budget": [23],
        "args": ["new source"],
    }
    # The narrower install is not an exemption from the full-generation check.
    assert main(["codegen", "--check"]) == 1
