"""The kernel library: laps, xwg, comm and folds.

Shared kernel parts written in the DSL: each part has a host-side
declaration (the arguments it adds and their memory) and a body-side object
whose calls emit statements through the Builder. The library uses only the
language's public surface and the package root, so a library
of one's own can do all this one does.

Importing ``thin.lib`` loads the DSL; the parts are its modules, and
``Laps`` and ``bf16_sum`` are here too.
"""

from thin.lib import comm, folds, laps, xwg
from thin.lib.folds import bf16_sum
from thin.lib.laps import Laps

__all__ = ["Laps", "bf16_sum", "comm", "folds", "laps", "xwg"]
