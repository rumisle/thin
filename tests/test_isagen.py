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
def test_a_clean_run_reproduces_the_manifest():
    rules = TARGETS["gfx1201"]
    isa = bundle.isa_root(Path(os.environ["THIN_ISA_BUNDLE"]), rules.member)
    assert (
        canonical.sha256(isa_tables.generate(isa, rules))
        == MANIFEST["isa_tables"]["sha256_canonical_json"]
    )
    rows = operand_table.generate(isa, rules)
    assert canonical.sha256(rows) == MANIFEST["operand_table"]["sha256_canonical_json"]


@pytest.mark.parametrize("table", ["isa_tables", "operand_table"])
def test_the_checked_in_tables_are_the_manifests(table):
    """The generator's checked-in output, without the bundle."""
    path = ROOT / "crates/thin-targets/data/gfx1201" / f"{table}.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == MANIFEST[table]["sha256_canonical_json"]
