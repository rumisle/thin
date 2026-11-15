"""Generate golden data from code, never reading the checked-in data as input."""

from __future__ import annotations

import argparse
import importlib
import json
import tempfile
from pathlib import Path

from . import ROOT, data_files, decode, encode, no_checked_data

COMPONENTS = (
    "target",
    "formats",
    "lib",
    "metron",
    "dsl",
    "runtime",
    "codegen",
    "opspec",
    "ir",
    "passes",
    "emu_text",
    "target_text",
)


def generators():
    """Explicit ownership, so unowned files cannot silently pass coverage."""
    with no_checked_data():
        return {name: importlib.import_module(f"tools.golden.{name}") for name in COMPONENTS}


def changes(before, after) -> list[str]:
    """Identify every changed JSON leaf; sequence indices are stable case ids."""
    found = []

    def visit(a, b, path):
        """Construct visit inputs."""
        if type(a) is not type(b):
            found.append(path)
        elif isinstance(a, dict):
            for key in sorted(a.keys() | b.keys()):
                at = f"{path}.{key}" if path else key
                if key not in a or key not in b:
                    found.append(at)
                else:
                    visit(a[key], b[key], at)
        elif isinstance(a, list):
            for i in range(max(len(a), len(b))):
                at = f"{path}[{i}]"
                if i >= len(a) or i >= len(b):
                    found.append(at)
                else:
                    visit(a[i], b[i], at)
        elif a != b:
            found.append(path)

    visit(before, after, "")
    return found


def main(argv=None) -> int:
    """Generate first, then compare or install; coverage is a separate strict check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=(*COMPONENTS, "all"), nargs="?", default="all")
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--coverage", action="store_true", help="fail on any data without a generator"
    )
    parser.add_argument(
        "--authored", action="store_true", help="select only authored source/program files"
    )
    parser.add_argument(
        "--authored-coverage", action="store_true", help="check authored-program ownership only"
    )
    parser.add_argument("--output", type=Path, help="alternate repository-shaped output root")
    parser.add_argument("--report", type=Path, help="write per-file changed JSON paths")
    args = parser.parse_args(argv)
    modules = generators()
    owners = {}
    for name, module in modules.items():
        for path in module.OUTPUTS:
            if path in owners:
                parser.error(f"two generators own {path}")
            owners[path] = name
    if args.coverage:
        missing = sorted(data_files() - owners.keys())
        stale = sorted(owners.keys() - data_files())
        print(json.dumps({"unowned": missing, "missing": stale}, indent=2))
        return bool(missing or stale)
    if args.authored_coverage:
        from .authored import coverage

        result = coverage(owners)
        print(json.dumps(result, indent=2))
        return int(any(result.values()))
    selected = modules if args.component == "all" else {args.component: modules[args.component]}
    if args.authored:
        from .authored import AUTHORED_FILES

        selected = {
            name: module
            for name, module in selected.items()
            if AUTHORED_FILES.intersection(module.OUTPUTS)
        }
    # --check generates a separate tree, not a hash of the checked-in data.
    with tempfile.TemporaryDirectory(prefix="thin-golden-") as temp:
        output = args.output or (Path(temp) if args.check else ROOT)
        report = {}
        for name, module in selected.items():
            with no_checked_data():
                fresh = module.generate()
            if set(fresh) != set(module.OUTPUTS):
                raise ValueError(f"{name}: generated files differ from declared ownership")
            for path, raw in fresh.items():
                if args.authored and path not in AUTHORED_FILES:
                    continue
                old_path = ROOT / path
                old = old_path.read_bytes() if old_path.exists() else None
                if args.authored:
                    from .authored import RECORDED_SECTIONS

                    recorded = RECORDED_SECTIONS.get(path, ())
                    if recorded:
                        if old is None:
                            raise ValueError(f"{path}: recorded sections require existing data")
                        before, after = decode(old, path), decode(raw, path)
                        # Fresh construction has finished. These tables are not
                        # inputs or generated expectations: retain recorded facts.
                        for section in recorded:
                            after[section] = before[section]
                        raw = encode(after, compressed=path.endswith(".gz"))
                if old is None:
                    diff = ["<new file>"]
                elif path.endswith((".json", ".json.gz")):
                    diff = changes(decode(old, path), decode(raw, path))
                else:
                    diff = [] if raw == old else ["<binary>"]
                report[path] = {"changed_fields": len(diff), "paths": diff}
                dest = output / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(raw)
                print(f"{path}: {len(diff)} changed fields")
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        unowned = sorted(data_files() - owners.keys()) if args.component == "all" else []
        if args.authored and args.component == "all":
            from .authored import coverage

            result = coverage(owners)
            unowned = sorted({path for paths in result.values() for path in paths})
        if args.check and unowned:
            print(f"Unowned data files: {len(unowned)} (run the scope's coverage check for paths)")
        return int(
            args.check and (bool(unowned) or any(x["changed_fields"] for x in report.values()))
        )


if __name__ == "__main__":
    raise SystemExit(main())
