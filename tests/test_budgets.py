"""The time-budget factor of a slower machine (tests/budgets.py)."""

import pytest

from tests import budgets


def test_without_the_variable_the_budgets_are_as_written():
    assert budgets.time_scale({}) == 1.0
    assert budgets.load(scale=1.0)["pytest"]["suite_seconds"] == 10


@pytest.mark.parametrize("raw", ["1", "2", "2.5", "4"])
def test_a_factor_from_one_to_the_maximum_is_read(raw):
    assert budgets.time_scale({budgets.SCALE_VARIABLE: raw}) == float(raw)


@pytest.mark.parametrize("raw", ["", "fast", "0", "0.5", "-2", "4.5", "inf", "nan"])
def test_an_unreadable_or_out_of_range_factor_is_an_error(raw):
    with pytest.raises(ValueError, match=budgets.SCALE_VARIABLE):
        budgets.time_scale({budgets.SCALE_VARIABLE: raw})


def test_the_factor_scales_time_and_never_memory():
    written, scaled = budgets.load(scale=1.0), budgets.load(scale=2.0)
    for name, section in written.items():
        for key, value in section.items():
            factor = 2.0 if key.endswith("seconds") else 1.0
            assert scaled[name][key] == value * factor, (name, key)
    assert scaled["pytest"]["peak_rss_mb"] == written["pytest"]["peak_rss_mb"]
