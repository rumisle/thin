"""The ISA table generator."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from tools.isagen import bundle, canonical, isa_tables, operand_table
from tools.isagen.targets import TARGETS

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads(
    (ROOT / "crates/thin-targets/tests/data/target/tables_manifest.json").read_text()
)


def test_the_pin_matches_the_manifest():
    assert (
        MANIFEST["bundle"]["file"],
        MANIFEST["bundle"]["sha256"],
    ) == (bundle.FILE, bundle.SHA256)


def test_another_bundle_is_refused(tmp_path):
    other = tmp_path / bundle.FILE
    other.write_bytes(b"not the pinned bundle")
    with pytest.raises(bundle.BundleError):
        bundle.check(other)


@pytest.mark.skipif("THIN_ISA_BUNDLE" not in os.environ, reason="set THIN_ISA_BUNDLE to the bundle")
@pytest.mark.parametrize("target", sorted(TARGETS))
def test_a_clean_run_reproduces_checked_in_tables(target):
    rules = TARGETS[target]
    isa = bundle.isa_root(Path(os.environ["THIN_ISA_BUNDLE"]), rules.member)
    data_dir = ROOT / "crates/thin-targets/data" / target
    got_isa = canonical.sha256(isa_tables.generate(isa, rules))
    got_ops = canonical.sha256(operand_table.generate(isa, rules))
    assert got_isa == hashlib.sha256((data_dir / "isa_tables.json").read_bytes()).hexdigest()
    assert got_ops == hashlib.sha256((data_dir / "operand_table.json").read_bytes()).hexdigest()


@pytest.mark.parametrize("table", ["isa_tables", "operand_table"])
def test_the_checked_in_tables_are_the_manifests(table):
    """The generator's checked-in output, without the bundle."""
    path = ROOT / "crates/thin-targets/data/gfx1201" / f"{table}.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == MANIFEST[table]["sha256_canonical_json"]


@pytest.mark.parametrize("target", sorted(TARGETS))
@pytest.mark.parametrize("table", ["isa_tables", "operand_table"])
def test_every_targets_checked_in_tables_are_canonical_json(target, table):
    path = ROOT / "crates/thin-targets/data" / target / f"{table}.json"
    parsed = json.loads(path.read_text())
    assert canonical.dumps(parsed).encode() == path.read_bytes()
