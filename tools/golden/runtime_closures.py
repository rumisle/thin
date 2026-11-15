"""kernel.json `closures`: generator modules written for this generator.

Each case's `src` becomes closure_gen.py in a fresh folder (with `files` beside
it), is imported, and `call` runs against generator_identity, as the replay
does. The identity or refusal is recorded from thin, never written here.
"""

from __future__ import annotations

NOTATION = (
    "Each `src` is written byte for byte (UTF-8) as closure_gen.py in a fresh folder, with "
    "`files` beside it; the folder goes first on the module path and closure_gen is imported. "
    "`call` then runs with the module's names plus thin.runtime's `generator_identity`. `ok` is "
    "r as [qualified name, sha256 of closure_gen.py's bytes, the generator's first line, its "
    "captures as [name, held text] pairs]; a module captured from `files` is held by that "
    "file's sha256. `raises`/`message`: the refusal."
)

CASES = [
    (
        "a captured helper function, held by its own identity and its captured lane count",
        """\
def make_family(lanes):
    def stride_of(wave):
        return wave * lanes

    def gen(WAVES: int = 4):
        return stride_of(WAVES)

    return gen


family = make_family(32)
""",
        {},
        "r = generator_identity(family)",
        "ok",
    ),
    (
        "three captured functions calling round a cycle: each cut at its first repeat by name",
        """\
def build_walkers():
    def red(k):
        return k > 0 and green(k - 1)

    def green(k):
        return k > 0 and blue(k - 1)

    def blue(k):
        return k <= 0 or red(k - 1)

    def gen(DEPTH: int = 9):
        return red(DEPTH)

    return gen


walker = build_walkers()
""",
        {},
        "r = generator_identity(walker)",
        "ok",
    ),
    (
        "a captured function that calls itself: a cycle of one",
        """\
def build_counter():
    def countdown(k):
        return 0 if k == 0 else countdown(k - 1)

    def gen(DEPTH: int = 3):
        return countdown(DEPTH)

    return gen


counter = build_counter()
""",
        {},
        "r = generator_identity(counter)",
        "ok",
    ),
    (
        "a captured local class and a captured module from a file beside it",
        """\
def build_tiles():
    import tile_consts

    class Layout:
        pad = 4

    def gen(TILE: int = 64):
        return tile_consts.ROWS * Layout.pad + TILE

    return gen


tiles = build_tiles()
""",
        {"tile_consts.py": "ROWS = 16\nCOLS = 8\n"},
        "r = generator_identity(tiles)",
        "ok",
    ),
    (
        "held captures in name order: a set sorted by text, a nested list as tuples, None",
        """\
def build_mixed(dims):
    flags = {"wide", "dense", "async"}
    spare = None

    def gen(N: int = 2):
        return (dims, flags, spare, N)

    return gen


mixed = build_mixed([2, [3, 4]])
""",
        {},
        "r = generator_identity(mixed)",
        "ok",
    ),
    (
        "a module-level generator captures nothing",
        """\
def gen_plain(BLOCK: int = 64):
    return BLOCK
""",
        {},
        "r = generator_identity(gen_plain)",
        "ok",
    ),
    (
        "a captured dict of modes",
        """\
def build_from(options):
    def gen(MODE: str = "fast"):
        return options[MODE]

    return gen


picked = build_from({"fast": 1, "slow": 2})
""",
        {},
        "r = generator_identity(picked)",
        "refusal",
    ),
    (
        "a captured list holding a dict: the refusal names the capture and the member",
        """\
def build_layers(layers):
    def gen(N: int = 1):
        return layers[N]

    return gen


stacked = build_layers([1, {"k": 3}])
""",
        {},
        "r = generator_identity(stacked)",
        "refusal",
    ),
    (
        "a captured builtin function",
        """\
def build_reducer():
    reduce_with = max

    def gen(N: int = 5):
        return reduce_with(N, 1)

    return gen


reducer = build_reducer()
""",
        {},
        "r = generator_identity(reducer)",
        "refusal",
    ),
    (
        "a captured function defined by exec",
        """\
def build_exec():
    scope = {}
    exec("def double(v):\\n    return 2 * v\\n", scope)
    double = scope["double"]

    def gen(N: int = 5):
        return double(N)

    return gen


doubled = build_exec()
""",
        {},
        "r = generator_identity(doubled)",
        "refusal",
    ),
    (
        "a captured instance of a plain class: its repr carries an address",
        """\
class Tuner:
    def __init__(self, level):
        self.level = level


def build_tuned():
    knob = Tuner(3)

    def gen(N: int = 5):
        return knob.level * N

    return gen


tuned = build_tuned()
""",
        {},
        "r = generator_identity(tuned)",
        "refusal",
    ),
    (
        "a captured bound method: its repr carries an address",
        """\
class Meter:
    def read(self):
        return 1


def build_metered():
    probe = Meter().read

    def gen(N: int = 5):
        return probe() + N

    return gen


metered = build_metered()
""",
        {},
        "r = generator_identity(metered)",
        "refusal",
    ),
    (
        "a captured name deleted before the generator escapes",
        """\
def build_forgetful():
    limit = 12

    def gen(N: int = 5):
        return min(N, limit)

    del limit
    return gen


forgetful = build_forgetful()
""",
        {},
        "r = generator_identity(forgetful)",
        "refusal",
    ),
    (
        "a captured run-time field value",
        """\
from thin.opspec import Field


def build_fielded():
    width = Field(24, 8, 64)

    def gen(N: int = 5):
        return width + N

    return gen


fielded = build_fielded()
""",
        {},
        "r = generator_identity(fielded)",
        "refusal",
    ),
]


def run_case(what, src, files, call, folder):
    """Write, import and call one case; returns its ok or raises record."""
    import importlib
    import sys

    from thin.runtime import generator_identity

    folder.mkdir(parents=True)
    (folder / "closure_gen.py").write_bytes(src.encode())
    for name, text in files.items():
        (folder / name).write_bytes(text.encode())
    modules = ("closure_gen", *(n[:-3] for n in files))
    sys.path.insert(0, str(folder))
    for gone in modules:
        sys.modules.pop(gone, None)
    try:
        scope = {**vars(importlib.import_module("closure_gen"))}
        scope["generator_identity"] = generator_identity
        try:
            exec(call, scope)
        except TypeError as e:
            return {"raises": "TypeError", "message": str(e)}
        r = scope["r"]
        return {"ok": [r[0], r[1], r[2], [list(p) for p in r[3]]]}
    finally:
        sys.path.remove(str(folder))
        for gone in modules:
            sys.modules.pop(gone, None)


def generate_section(temp) -> dict:
    """Evaluate each newly authored closure source in isolation."""
    cases = []
    for i, (what, src, files, call, kind) in enumerate(CASES):
        case = {"what": what, "src": src}
        if files:
            case["files"] = files
        case["call"] = call
        result = run_case(what, src, files, call, temp / f"closure{i}")
        if ("raises" in result) != (kind == "refusal"):
            raise AssertionError(f"{what}: planned {kind}, got {result}")
        case.update(result)
        cases.append(case)
    return {"notation": NOTATION, "cases": cases}
