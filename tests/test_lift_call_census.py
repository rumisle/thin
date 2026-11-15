"""Report aggregation preserves site counts rather than comparing first errors."""

from __future__ import annotations

import copy
import sys

import pytest

from tools.lift_call_census import COUNT_MAPS, COUNTERS, EXAMPLES, main, merge


def _shard(count, example):
    result = {key: count for key in COUNTERS}
    result.update({key: {"count": count} for key in COUNT_MAPS})
    result.update({key: {"site": example} for key in EXAMPLES})
    result["edge_set_sizes"] = {"return": {"300": count}}
    result["refusals"] = [
        {"class": "policy", "sites": count * 2, "kernels": count, "example": example}
    ]
    return result


def test_census_shards_sum_disjoint_counts_without_mutating_their_inputs():
    shards = [_shard(2, "first"), _shard(3, "second")]
    original = copy.deepcopy(shards)
    report = merge(shards)
    assert shards == original
    assert all(report[key] == 5 for key in COUNTERS)
    assert all(report[key] == {"count": 5} for key in COUNT_MAPS)
    assert all(report[key] == {"site": "first"} for key in EXAMPLES)
    assert report["edge_set_sizes"] == {"return": {"300": 5}}
    assert report["refusals"] == [
        {"class": "policy", "sites": 10, "kernels": 5, "example": "first"}
    ]


def test_census_rejects_duplicate_object_paths_before_running_a_shard(monkeypatch, tmp_path):
    path = str(tmp_path / "one.co")
    monkeypatch.setattr(sys, "argv", ["census", "--json", str(tmp_path / "out"), path, path])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


@pytest.mark.parametrize("jobs", ["0", "-1"])
def test_census_rejects_nonpositive_worker_counts(monkeypatch, tmp_path, jobs):
    monkeypatch.setattr(
        sys,
        "argv",
        ["census", "--json", str(tmp_path / "out"), "--jobs", jobs, "one.co"],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
