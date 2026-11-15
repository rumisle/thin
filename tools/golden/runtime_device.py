"""Device and allocation scenarios executed through the public runtime API.

Each step specifies a call and its intended success or refusal class.
"""

from __future__ import annotations

import json
import struct

NOTATION = (
    "Byte-level device calls, no target needed. Names in scope: `runtime` (thin.runtime), the "
    "dtypes `F32`, `BF16`, `U8`, and `f32(*xs)`, the little-endian f32 bytes of xs. `steps`: "
    "`steps_setup` makes one emulated device `emu` with a 48-byte f32 ramp and a 48-byte filled "
    "sink; every step's `call` then runs in that namespace in list order (an earlier step's "
    "refusal does not stop the next). `ok` is the call's `r`: bytes as hex, null for None and "
    "for a group that was made; else `raises`/`message`. `guards`: `guards_setup` writes a "
    "host-memory device `h` against the device-implementer protocol, keeping the default guard. "
    "Its stand-in kernel reads a 16-byte kernarg (`poke_args`: a buffer's pointer, its size, then "
    "how many bytes to scribble past its end and before its start) and writes those bytes "
    "outside the buffer. Guards run in order in one namespace; `ok` is `r`; a guard fault also "
    "records `cause`, and `synced` is the last text the sync hook received (launch names joined "
    "by '; ', or 'the stream')."
)

STEPS_SETUP = """\
emu = runtime.EmulatedDevice()
ramp = emu.tensor(F32, f32(*(2.5 - 0.75 * i for i in range(12))))
sink = emu.tensor(F32, nbytes=48, fill=0x3C)"""

STEPS = [
    ("read back a tensor made from twelve f32 values", "r = emu.read(ramp)", "ok"),
    (
        "read the third 16-byte window of the ramp",
        "r = emu.read(emu.slice(ramp, 32, 16))",
        "ok",
    ),
    (
        "an empty slice at the very end",
        "r = emu.read(emu.slice(ramp, 48, 0))",
        "ok",
    ),
    ("a tensor made by nbytes holds its fill byte", "r = emu.read(sink)", "ok"),
    (
        "copy the ramp's first window into the sink's second, fill its third with 0xA5",
        "emu.copy(emu.slice(sink, 16, 16), emu.slice(ramp, 0, 16))\n"
        "emu.fill(emu.slice(sink, 32, 16), 0xA5)\n"
        "r = emu.read(sink)",
        "ok",
    ),
    (
        "a bf16 tensor of no bytes",
        "r = emu.read(emu.tensor(BF16, nbytes=0))",
        "ok",
    ),
    (
        "data given with nbytes",
        "r = emu.tensor(U8, bytes(3), nbytes=3)",
        "refusal",
    ),
    (
        "only a fill: neither data nor nbytes",
        "r = emu.tensor(BF16, fill=7)",
        "refusal",
    ),
    ("ten bytes cannot be f32 elements", "r = emu.tensor(F32, nbytes=10)", "refusal"),
    (
        "five bytes cannot be bf16 elements",
        "r = emu.tensor(BF16, nbytes=5)",
        "refusal",
    ),
    (
        "three bytes of data cannot be bf16 elements",
        'r = emu.tensor(BF16, b"\\x80\\x3f\\x00")',
        "refusal",
    ),
    (
        "a negative size that is still a multiple of the element",
        "r = emu.tensor(F32, nbytes=-8)",
        "refusal",
    ),
    ("nbytes a float", "r = emu.tensor(U8, nbytes=8.0)", "refusal"),
    (
        "a window running past the end",
        "r = emu.slice(ramp, 32, 32)",
        "refusal",
    ),
    ("a window at offset 8", "r = emu.slice(ramp, 8, 8)", "refusal"),
    ("a window at a negative offset", "r = emu.slice(ramp, -16, 16)", "refusal"),
    ("a window of 6 bytes of f32", "r = emu.slice(ramp, 0, 6)", "refusal"),
    (
        "copy a 32-byte window into the 48-byte sink",
        "r = emu.copy(sink, emu.slice(ramp, 16, 32))",
        "refusal",
    ),
    (
        "fill with the first value past a byte",
        "r = emu.fill(sink, 1 << 8)",
        "refusal",
    ),
    ("fill with -1", "r = emu.fill(sink, -1)", "refusal"),
    ("fill with a bool", "r = emu.fill(ramp, True)", "refusal"),
    (
        "a fresh device asked to fill this device's sink",
        "r = runtime.EmulatedDevice().fill(sink, 0)",
        "refusal",
    ),
    (
        "plain bytes given where a buffer belongs",
        'r = emu.read(b"\\x00" * 16)',
        "refusal",
    ),
    ("close the device", "r = emu.close()", "ok"),
    (
        "a tensor asked of the closed device",
        "r = emu.tensor(U8, nbytes=16)",
        "refusal",
    ),
    ("a read on the closed device", "r = emu.read(sink)", "refusal"),
    ("a second close", "r = emu.close()", "refusal"),
    ("a group of world=0", "r = runtime.emulated_group(world=0)", "refusal"),
    ("a group of -3 ranks", "r = runtime.emulated_group(-3)", "refusal"),
    (
        "a group with two unknown interleaving keys beside a known one",
        "r = runtime.emulated_group(3, jitter=0, warp=2, lanes=1)",
        "refusal",
    ),
    (
        "a group of no stall rounds",
        "r = runtime.emulated_group(4, stall_rounds=0)",
        "refusal",
    ),
    (
        "a group of 2.5 stall rounds",
        "r = runtime.emulated_group(2, stall_rounds=2.5)",
        "refusal",
    ),
    ("a group of 2.0 ranks", "r = runtime.emulated_group(2.0)", "refusal"),
    ("a group of True ranks", "r = runtime.emulated_group(True)", "refusal"),
    (
        "a group of two ranks with a known interleaving key is made",
        "r = runtime.emulated_group(2, speed=[1, 3])",
        "ok",
    ),
]

GUARDS_SETUP = """\
class Host(runtime.Device):
    def __init__(self):
        super().__init__()
        self.synced, self.runs, self.blocks = [], [], {}
        self.next_base = 0x7A0000
    def alloc_bytes(self, n, fill, meta):
        base = self.next_base
        self.next_base += 0x10000
        self.blocks[base] = bytearray([fill]) * n
        return base, base
    def write_bytes(self, handle, offset, data):
        self.blocks[handle][offset:offset + len(data)] = data
    def read_bytes(self, handle, offset, n):
        return bytes(self.blocks[handle][offset:offset + n])
    def copy_bytes(self, dst, dst_offset, src, src_offset, n):
        self.blocks[dst][dst_offset:dst_offset + n] = self.blocks[src][src_offset:src_offset + n]
    def fill_bytes(self, handle, offset, n, byte):
        self.blocks[handle][offset:offset + n] = bytes([byte]) * n
    def sync(self, what):
        self.synced.append(what)
    def free(self):
        self.blocks.clear()
    def load(self, image, code_object, asm, block, key):
        return ("loaded", key)
    def run(self, code, key, grid, block, kernarg, what, cooperative):
        ptr = int.from_bytes(kernarg[0:8], "little")
        n = int.from_bytes(kernarg[8:12], "little")
        past, before = kernarg[12], kernarg[13]
        base = max(b for b in self.blocks if b <= ptr)
        mem, at = self.blocks[base], ptr - base
        mem[at + n:at + n + past] = b"\\xee" * past
        mem[at - before:at] = b"\\xee" * before
        self.runs.append([what, list(grid), cooperative])
def poke_args(t, past=0, before=0):
    return (
        t.ptr.to_bytes(8, "little") + t.nbytes.to_bytes(4, "little")
        + bytes([past, before, 0, 0])
    )
h = Host()
lhs = h.tensor(F32, nbytes=16)
rhs = h.tensor(BF16, f32(1.5, -2.0, 0.25, 8.0))
acc = h.tensor(U8, nbytes=32, fill=9)
prog = runtime.Program(h, "poke", (64, 1), 16, h.load(b"", None, "", (64, 1), "poke"))
far = Host().tensor(U8, nbytes=48)"""

GUARDS = [
    (
        "the guard width, its fill, and pointers past each allocation's guard",
        'r = {"guard": h.guard, "fill": runtime.GUARD_FILL, "lhs": hex(lhs.ptr), '
        '"rhs": hex(rhs.ptr), "window": hex(h.slice(acc, 16, 16).ptr), '
        '"head": h.blocks[0x7A0000][:4].hex()}',
        "ok",
    ),
    (
        "three clean launches, one name twice, one cooperative, then a wait",
        'runtime.dispatch(prog, (4, 1, 1), poke_args(lhs), {"lhs": lhs}, what="scale")\n'
        'runtime.dispatch(prog, (2, 2, 1), poke_args(rhs), {"lhs": lhs, "rhs": rhs}, '
        'what="shift")\n'
        'runtime.dispatch(prog, (1, 1, 1), poke_args(lhs), {"lhs": lhs}, what="scale", '
        "cooperative=True)\n"
        "h.wait()\n"
        'r = {"synced": h.synced[-1], "runs": h.runs[-3:]}',
        "ok",
    ),
    (
        "a launch scribbles 2 bytes past acc, a later one reads acc; wait",
        'runtime.dispatch(prog, (1, 1, 1), poke_args(acc, past=2), {"acc": acc}, what="spill")\n'
        'runtime.dispatch(prog, (3, 1, 1), poke_args(acc), {"acc": acc, "lhs": lhs}, '
        'what="reader")\n'
        "h.wait()",
        "refusal",
    ),
    (
        "an unnamed launch scribbles 1 byte before lhs; wait: the fault also names reader, "
        "whose check of lhs the previous fault cut short",
        'runtime.dispatch(prog, (1, 1, 1), poke_args(lhs, before=1), {"lhs": lhs})\nh.wait()',
        "refusal",
    ),
    (
        "a kernarg 4 bytes short",
        'runtime.dispatch(prog, (1, 1, 1), poke_args(rhs)[:12], {"rhs": rhs}, what="short")',
        "refusal",
    ),
    (
        "a buffer of a second host device",
        'runtime.dispatch(prog, (1, 1, 1), poke_args(rhs), {"rhs": rhs, "far": far})',
        "refusal",
    ),
    (
        "a wait after the faults with nothing enqueued",
        'h.synced.clear()\nh.wait()\nr = {"synced": h.synced}',
        "ok",
    ),
    (
        "program() of 32 counting bytes",
        'r = runtime.program(h, bytes(range(32)), "", "probe")',
        "refusal",
    ),
    (
        "program() of an ELF64 header for machine 0",
        'r = runtime.program(h, b"\\x7fELF\\x02\\x01" + bytes(58), "", "probe")',
        "refusal",
    ),
    (
        "program() of junk without a key",
        'r = runtime.program(h, b"\\x00\\xff" * 8, "")',
        "refusal",
    ),
    (
        "a dispatch on the closed device",
        'h.close()\nruntime.dispatch(prog, (1, 1, 1), poke_args(lhs), {"lhs": lhs}, what="late")',
        "refusal",
    ),
]


def namespace() -> dict:
    """The names the device notation binds (the replay binds the same)."""
    import thin.runtime as runtime
    from thin import formats

    def f32(*xs):
        """Construct f32 inputs."""
        return struct.pack(f"<{len(xs)}f", *xs)

    d = formats.DTYPES
    return {"runtime": runtime, "F32": d["f32"], "BF16": d["bf16"], "U8": d["u8"], "f32": f32}


def _outcome(scope, what, call, kind, *, guards):
    """Run one call where the earlier calls ran; record what it gives."""
    import builtins

    from thin.runtime import errors

    case = {"what": what, "call": call}
    scope.pop("r", None)
    try:
        exec(call, scope)
    except Exception as e:  # recorded refusals: exact class, message and cause
        name = type(e).__name__
        if (getattr(errors, name, None) or getattr(builtins, name, None)) is not type(e):
            raise AssertionError(f"{what}: {name} is not nameable by the replay") from e
        case["raises"], case["message"] = name, e.args[0]
        if isinstance(e, errors.DeviceFault):
            case["cause"] = e.cause
            if guards:
                case["synced"] = scope["h"].synced[-1]
    else:
        r = scope.get("r")
        if isinstance(r, bytes):
            r = r.hex()
        elif isinstance(r, list):
            r = None
        copied = json.loads(json.dumps(r))  # a copy: later calls may change r's lists
        if copied != r:
            raise AssertionError(f"{what}: r {r!r} has no exact JSON form")
        case["ok"] = copied
    got = "refusal" if "raises" in case else "ok"
    if got != kind:
        raise AssertionError(f"{what}: planned {kind}, got {case}")
    return case


def _section(setup, plan, *, guards):
    scope = namespace()
    exec(setup, scope)
    return [_outcome(scope, w, c, k, guards=guards) for w, c, k in plan]


def generate_doc() -> dict:
    """Construct this document and record actual API outcomes."""
    return {
        "notation": NOTATION,
        "steps_setup": STEPS_SETUP,
        "steps": _section(STEPS_SETUP, STEPS, guards=False),
        "guards_setup": GUARDS_SETUP,
        "guards": _section(GUARDS_SETUP, GUARDS, guards=True),
    }
