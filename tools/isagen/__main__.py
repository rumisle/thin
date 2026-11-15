"""The generator's command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.isagen import bundle, canonical, isa_tables, operand_table
from tools.isagen.targets import TARGETS

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    """Generates a target's tables into its crate's data directory."""
    parser = argparse.ArgumentParser(prog="python -m tools.isagen", description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help=f"the pinned {bundle.FILE}")
    parser.add_argument("--target", choices=sorted(TARGETS), required=True)
    args = parser.parse_args(argv)
    rules = TARGETS[args.target]
    isa = bundle.isa_root(args.bundle, rules.member)
    out = ROOT / "crates" / "thin-targets" / "data" / args.target
    out.mkdir(parents=True, exist_ok=True)
    for name, table in [
        ("isa_tables.json", isa_tables.generate(isa, rules)),
        ("operand_table.json", operand_table.generate(isa, rules)),
    ]:
        (out / name).write_text(canonical.dumps(table), encoding="ascii")
        print(f"{out / name}: sha256 {canonical.sha256(table)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
