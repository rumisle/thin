"""Evaluate independently constructed DSL snippets into golden records."""

from __future__ import annotations

import re

from tests import notation
from thin import codegen


def error_record(error, *, rendering=False):
    """Preserve exact native refusals, normalizing only unstable source positions."""
    message = getattr(error, "msg", error.args[0] if error.args else "")
    if rendering:
        message = re.sub(r"^(xwg|comm|laps|folds)\.py:\d+:", r"<\1>:", str(error).split("\n")[0])
    return {"class": type(error).__name__, "message": message}


def evaluate(name, source, *, library=False):
    """Build, lower and render a case using thin, without reading a fixture."""
    case = {"name": name, "src": source}
    try:
        builder = notation.run_snippet(source, library=library)
    except Exception as error:  # invalid inputs are intentional constructor cases
        case["error"] = error_record(error)
        return case
    case.update(built=notation.built(builder), values=notation.values(builder))
    records = notation.records(builder)
    if records:
        case["records"] = records
    try:
        case["render"] = codegen.render(builder)
    except Exception as error:
        case["render_error"] = error_record(error, rendering=True)
    return case
