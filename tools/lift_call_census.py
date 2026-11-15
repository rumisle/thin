"""Run disjoint call-census shards through the shared native corpus walker.

Build first: cargo build --release -p thin-lift --example lift_census
Run: python -m tools.lift_call_census --json counts.json PATH...
Only counts and examples are merged; no corpus bytes enter the report.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COUNTERS = (
    "attempted",
    "converted",
    "programs_with_indirect",
    "resolved_programs",
    "unresolved_sites",
    "dead_kernels_before",
    "dead_kernels_after",
)
COUNT_MAPS = (
    "corpus",
    "checks",
    "transfers",
    "dead_sites",
    "dead_classes",
    "dead_word_classes",
    "unassign",
)
EXAMPLES = ("transfer_examples", "dead_examples", "unassign_examples")


def add_counts(destination: dict, source: dict) -> None:
    """Sum counters from disjoint kernel sets."""
    for key, count in source.items():
        destination[key] = destination.get(key, 0) + count


def merge(shards: list[dict]) -> dict:
    """Preserve first-shard examples and sum site and kernel counts."""
    result = {key: 0 for key in COUNTERS}
    result.update({key: {} for key in (*COUNT_MAPS, *EXAMPLES, "edge_set_sizes")})
    refusals = {}
    for shard in shards:
        for key in COUNTERS:
            result[key] += shard[key]
        for key in COUNT_MAPS:
            add_counts(result[key], shard[key])
        for key in EXAMPLES:
            for name, example in shard[key].items():
                result[key].setdefault(name, example)
        for kind, sizes in shard["edge_set_sizes"].items():
            add_counts(result["edge_set_sizes"].setdefault(kind, {}), sizes)
        for finding in shard["refusals"]:
            row = refusals.setdefault(finding["class"], dict(finding, sites=0, kernels=0))
            row["sites"] += finding["sites"]
            row["kernels"] += finding["kernels"]
    result["refusals"] = list(refusals.values())
    return result


def main() -> None:
    """Run bounded native subprocesses, then write one structured report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/examples/lift_census")
    parser.add_argument("objects", type=Path, nargs="+")
    args = parser.parse_args()
    if args.jobs < 1 or args.timeout <= 0:
        parser.error("jobs and timeout must be positive")
    paths = sorted({str(path.resolve()) for path in args.objects})
    if len(paths) != len(args.objects):
        parser.error("duplicate object paths would double-count kernels")
    jobs = min(args.jobs, len(paths))
    binary = str(args.binary.resolve())
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="thin-call-census-") as directory:

        def run(index: int) -> dict:
            output = Path(directory) / f"{index}.json"
            command = [binary, "--physical-json", str(output), *paths[index::jobs]]
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=args.timeout, cwd=ROOT
            )
            if completed.returncode:
                raise RuntimeError(
                    f"census shard {index} failed:\n{completed.stderr}\n{completed.stdout[-6000:]}"
                )
            print(f"shard {index}: {time.monotonic() - started:.1f}s", flush=True)
            return json.loads(output.read_text())

        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            result = merge(list(pool.map(run, range(jobs))))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result["method"] = {
        "source_commit": revision,
        "workers": jobs,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "input": "disjoint round-robin shards over the shared native corpus walker",
        "dead_findings": "All instruction sites, not differences between first diagnostics.",
    }
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"report: {args.json}", flush=True)


if __name__ == "__main__":
    main()
