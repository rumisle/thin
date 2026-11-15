"""spec.json: spec names, field declarations, ops, tensors, given tensors and group ops.

Every text here is a scenario of the op layer's public API; outcomes are
recorded by ``tools.golden.opspec`` replaying the texts.
"""

from __future__ import annotations

from tools.golden.opspec import case

FILE = "<test>/opspec/spec.py"

SETUP = '''\
def kind_of(v):
    """[type name, value] of a field value; a Field also shows the range it serves."""
    if isinstance(v, Field):
        return ["Field", int(v), *v.range]
    return [type(v).__name__, v]

class Norm(Spec, name="decl.norm"):
    fmt = Choice("bf16", "f16")
    rows = Int(lo=2, hi=64)
    eps_mode = Choice("fast", "exact", default="exact")
    width = Int(default=8, lo=2, hi=32)
    lanes = Choice(32, 64, 16, default=32)
    fused = Bool(default=True)
    def signature(op):
        return {"a": In(op.fmt, op.rows), "b": Out(op.fmt, op.rows)}

def summary(op):
    return {"key": op.key(), "values": {n: kind_of(v) for n, v in op.field_values().items()},
            "given": sorted(op.tensors), "group": op.group, "rank": op.rank}

def facts(t):
    return [t.io, t.dtype.name, [int(s) for s in t.shape], t.shared]

def bits(t):
    numpy_name, shape = t.storage()
    p = t.poisoned()
    raw = p.reshape(-1).view(np.uint8).tobytes()
    return {"numpy": numpy_name, "shape": list(shape), "torch": t.dtype.torch,
            "poison": {"dtype": p.dtype.name, "shape": list(p.shape),
                       "bytes": len(raw), "values": sorted(set(raw))}}

class Pair(Spec, name="decl.pair"):
    cols = Int(lo=1)
    kind = Choice("bf16", "f32", "e2m1", default="bf16")
    def signature(op):
        return {"src": In(op.kind, 2, op.cols), "dst": Out(op.kind, 2, op.cols)}

def offer(fields, **tensors):
    op = Pair(**fields, **tensors)
    kept = [n for n, v in tensors.items() if getattr(op, n) is v]
    return {"given": sorted(op.tensors), "kept": kept}

class Overlap(Spec, name="decl.overlap"):
    a = Int(lo=1)
    b = Int(lo=1)
    def signature(op):
        return {"b": Out("u8", op.a), "a": In("u8", op.b), "c": In("u8", 1)}

class Gather(Spec, name="decl.gather", group="size"):
    size = Int(lo=2, hi=6)
    part = Int(lo=1)
    def signature(op):
        return {"piece": In("f32", op.part), "whole": Out("f32", op.size, op.part, shared=True)}

class Lonely(Spec, name="decl.lonely"):
    m = Int(lo=1)
    def signature(op):
        return {"buf": InOut("u8", op.m, shared=True)}

crew = group.emulated(3)
pair = group.emulated(2)
whole = [h.tensor("whole", "f32", (3, 2)) for h in crew]
foreign = pair[0].tensor("whole", "f32", (3, 2))

def placed(op):
    return {"key": op.key(), "group": None if op.group is None else repr(op.group), "rank": op.rank,
            "given": sorted(op.tensors)}
'''

NOTATION = (
    "Executable: `setup` runs first, then every `call` in document order, in one namespace "
    "holding thin.opspec's exported names, thin.runtime's LaunchError, DeviceFault and Bound, "
    "np (numpy) and torch; texts are compiled with the file name `file`. A call sets r: ok is r "
    "as JSON holds it; refused/message are the exception's class and text, (<file>) masking a "
    "source path. kind_of(v) is [type name, value]; facts(t) is [io, dtype, shape, shared]; "
    "bits(t) is a tensor's storage and its poisoned array; offer(fields, **tensors) makes a "
    "decl.pair op and says which given tensors it holds as passed; placed(op) is a group op's "
    "key, handle and given tensors."
)


def statement(cls, head, body="pass"):
    """A class statement then r = its name()."""
    return f"class {cls}(Spec{head}):\n    {body}\nr = {cls}.name()"


def names():
    """Spec names and group fields, valid and refused, and one name one spec."""
    plans = [
        ("no name at all", "Gelu", ""),
        ("a name of one part", "Gelu", ", name='gelu'"),
        ("a namespace and a name", "Gelu", ", name='decl.gelu'"),
        (
            "four parts, underscores and digits after the first character",
            "Tanh",
            ", name='decl._fused.gelu_tanh2.v1'",
        ),
        ("a part beginning with a digit", "Conv", ", name='decl.1x1conv'"),
        ("an empty last part", "Gelu", ", name='decl.gelu.'"),
        ("an empty first part", "Gelu", ", name='.decl.gelu'"),
        ("a space inside a part", "Gelu", ", name='decl.gelu tanh'"),
        ("a non-ASCII letter", "Gelu", ", name='decl.g\u00e9lu'"),
        ("a tuple of parts", "Gelu", ", name=('decl', 'gelu')"),
    ]
    out = [case(statement(cls, head), case_=desc) for desc, cls, head in plans]
    out.append(
        case(
            "class Ring(Spec, name='decl.ring', group='peers'):\n"
            "    peers = Int(lo=1, hi=16)\n"
            "    n = Int(lo=1)\n"
            "r = [Ring.name(), Ring.group_field(), list(Ring.fields())]",
            case_="group names an Int field",
        )
    )
    out += [
        case(
            statement("Flag", ", name='decl.flag', group='peers'", "peers = Bool()"),
            case_="group names a Bool field (a Choice)",
        ),
        case(
            statement("Lost", ", name='decl.lost', group='peers'", "n = Int(lo=1)"),
            case_="group names a field the spec does not have",
        ),
        case(
            statement("Anon", ", group='peers'"),
            case_="the group is checked before the name: no field, no name",
        ),
        case(
            statement("Anon", ", name='decl', group='peers'", "peers = Int(lo=2)"),
            case_="a group that holds, then a name that does not",
        ),
        case(
            "exec(compile(\"class Copy(Spec, name='decl.gelu'):\\n    pass\\n\", "
            '"<test>/opspec/elsewhere_too.py", "exec"), {"Spec": Spec})',
            case_="a name taken by a class statement in another file",
            effect=True,
        ),
        case(
            statement("Twin", ", name='decl.gelu'"),
            case_="a name taken by another class statement in the same file",
        ),
        case(
            "code = compile(\"class Again(Spec, name='decl.again'):\\n    width = Int(lo=1)\\n"
            '    def signature(op): return {}\\n", '
            '"<test>/opspec/again.py", "exec")\n'
            'first, second = {"Spec": Spec, "Int": Int}, {"Spec": Spec, "Int": Int}\n'
            "exec(code, first)\n"
            "exec(code, second)\n"
            'r = [first["Again"] is second["Again"], second["Again"].name(), '
            'second["Again"](width=3).key()]',
            case_="one class statement run twice: two classes, one name",
        ),
    ]
    return out


def declarations():
    """Choice, Bool and Int declarations on their own."""
    rows = [
        ("no options", "r = Choice()"),
        ("a float and bytes among ints", "r = Choice(3, 2.5, b'q')"),
        ("an int twice", "r = Choice(8, 16, 8)"),
        (
            "1/True and 0/False are four options",
            "r = list(Choice(1, True, 0, False).options)",
        ),
        (
            "a default not among the options",
            "r = Choice('bf16', 'f16', default='f32')",
        ),
        (
            "default 0 against False",
            "r = Choice(False, True, default=0)",
        ),
        (
            "Bool's options and default",
            "b = Bool()\nr = {'class': type(b).__name__, 'options': list(b.options), "
            "'default': b.default}",
        ),
        ("a False default is kept", "r = Bool(default=False).default"),
        ("lo above hi", "r = Int(lo=0, hi=-1)"),
        ("a float lo", "r = Int(lo=1.5)"),
        ("a bool hi", "r = Int(hi=True)"),
        ("a default below lo", "r = Int(default=0, lo=1)"),
        (
            "a field with no bounds",
            "f = Int()\nr = [f.default, f.lo, f.hi, repr(f)]",
        ),
    ]
    return [case(text, case_=desc) for desc, text in rows]


def ops():
    """decl.norm ops: defaults, every field, accepted and refused values."""
    rows = [
        ("only the required fields", "fmt='f16', rows=12"),
        (
            "every field at its upper end or last option",
            "fmt='f16', rows=64, eps_mode='fast', width=32, lanes=16, fused=False",
        ),
        ("every Int at its lower end", "fmt='bf16', rows=2, width=2"),
        ("a numpy int32 for an Int", "fmt='bf16', rows=np.int32(9)"),
        (
            "a Field for an Int is held as given",
            "fmt='bf16', rows=Field(10, 2, 64)",
        ),
        ("the required Int missing", "fmt='f16'"),
        ("nothing given: the first declared is named", ""),
        ("an Int one below lo", "fmt='f16', rows=1"),
        ("an Int one above hi", "fmt='f16', rows=65"),
        ("a defaulted Int above hi", "fmt='f16', rows=4, width=33"),
        ("True for an Int", "fmt='f16', rows=True"),
        ("a float for an Int", "fmt='f16', rows=8.0"),
        ("a str for an Int", "fmt='f16', rows='8'"),
        ("a str that is no option", "fmt='f32', rows=8"),
        ("0 for a Bool", "fmt='f16', rows=8, fused=0"),
        (
            "a numpy int for an int Choice",
            "fmt='f16', rows=8, lanes=np.int64(64)",
        ),
        ("None for a defaulted field", "fmt='f16', rows=8, eps_mode=None"),
        ("two unknown keywords", "fmt='f16', rows=8, gamma=1, beta=2"),
        (
            "a tensor given with the fields",
            "fmt='f16', rows=6, a=np.zeros(6, np.uint16)",
        ),
    ]
    out = [case("r = list(Norm.fields())", case_="field order is declaration order")]
    out += [case(f"r = summary(Norm({kw}))", case_=desc) for desc, kw in rows]
    return out


def made():
    """Signature tensors as In/Out/InOut make them."""
    rows = [
        ("an Out of two sizes", 'Out("f16", 3, 5)'),
        ("an InOut of one size", 'InOut("i8", 7)'),
        ("a shared In", 'In("e4m3fn", 2, 2, shared=True)'),
        ("no sizes", 'Out("u64")'),
        ("a zero size", 'InOut("f32", 4, 0)'),
        ("a negative size, kept", 'In("u8", -3)'),
        ("an unknown dtype", 'Out("bf8", 2)'),
        ("the shape as one list", 'In("f16", [4, 4])'),
        ("a bool size", 'In("u32", 2, False)'),
        ("a float size", 'In("u32", 10.0)'),
        ("a numpy size", 'In("u8", np.uint16(5))'),
        ("a dtype record, not a name", 'Out(In("bf16", 1).dtype, 2)'),
    ]
    return [case(f"r = facts({expr})", case_=desc) for desc, expr in rows]


def storage():
    """Storage and poisoned arrays of every dtype, sub-byte packing included."""
    plans = [
        ("f32", (2, 2)),
        ("bf16", (3,)),
        ("f16", (1, 4)),
        ("e4m3fn", (5,)),
        ("e5m2", (2, 3)),
        ("e8m0", (4,)),
        ("e2m1", (2, 4)),
        ("e2m1", (0,)),
        ("e2m1", (5,)),
        ("e2m1", ()),
        ("e2m3", (2,)),
        ("e2m3", ()),
        ("e3m2", (4,)),
        ("u8", (1,)),
        ("i8", (3,)),
        ("u32", (1, 3)),
        ("u32", ()),
        ("u64", (2,)),
    ]
    out = []
    for dtype, shape in plans:
        sizes = "".join(f", {s}" for s in shape)
        out.append(case(f"r = bits(Out({dtype!r}{sizes}))", dtype=dtype, shape=list(shape)))
    return out


def given():
    """Tensors given to a decl.pair op: kind, dtype, shape, contiguity, place, check order."""
    rows = [
        (
            "bf16 held as uint16",
            "{'cols': 6}",
            "src=np.zeros((2, 6), np.uint16)",
        ),
        (
            "both tensors",
            "{'cols': 3, 'kind': 'f32'}",
            "src=np.ones((2, 3), np.float32), dst=np.empty((2, 3), np.float32)",
        ),
        (
            "f32 given as int32",
            "{'cols': 3, 'kind': 'f32'}",
            "dst=np.zeros((2, 3), np.int32)",
        ),
        (
            "f32 as float32",
            "{'cols': 1, 'kind': 'f32'}",
            "src=np.zeros((2, 1), np.float32)",
        ),
        (
            "bf16 given as float16",
            "{'cols': 6}",
            "src=np.zeros((2, 6), np.float16)",
        ),
        (
            "e2m1 two a byte",
            "{'cols': 6, 'kind': 'e2m1'}",
            "dst=np.zeros((2, 3), np.uint8)",
        ),
        (
            "e2m1 a byte each",
            "{'cols': 6, 'kind': 'e2m1'}",
            "dst=np.zeros((2, 6), np.uint8)",
        ),
        (
            "one column too many",
            "{'cols': 6}",
            "src=np.zeros((2, 7), np.uint16)",
        ),
        ("the right size, flat", "{'cols': 6}", "src=np.zeros(12, np.uint16)"),
        (
            "column-major order",
            "{'cols': 6}",
            "src=np.zeros((2, 6), np.uint16, order='F')",
        ),
        ("a tuple of numbers", "{'cols': 2}", "src=((0, 0), (0, 0))"),
        (
            "a CPU torch float32",
            "{'cols': 4, 'kind': 'f32'}",
            "src=torch.zeros(2, 4)",
        ),
        (
            "a CPU torch bfloat16",
            "{'cols': 4}",
            "dst=torch.zeros(2, 4, dtype=torch.bfloat16)",
        ),
        (
            "a torch float16 for bf16",
            "{'cols': 4}",
            "dst=torch.zeros(2, 4, dtype=torch.float16)",
        ),
        (
            "a transposed torch view",
            "{'cols': 2, 'kind': 'f32'}",
            "src=torch.zeros(2, 2).t()",
        ),
        (
            "torch and numpy together",
            "{'cols': 4, 'kind': 'f32'}",
            "src=torch.zeros(2, 4), dst=np.zeros((2, 4), np.float32)",
        ),
        (
            "the wrong dtype and shape: dtype named",
            "{'cols': 6}",
            "src=np.zeros((3, 3), np.float32)",
        ),
        (
            "the wrong shape, column-major: shape named",
            "{'cols': 6}",
            "src=np.zeros((2, 5), np.uint16, order='F')",
        ),
    ]
    return [case(f"r = offer({fields}, {kw})", case_=desc) for desc, fields, kw in rows]


def misc():
    """Signature/field clashes, attribute access, with_tensors and subclassing."""
    return [
        case("r = Overlap(a=2, b=3)", case_="two signature tensors named as fields"),
        case("r = Pair(cols=2).bias", case_="neither a field nor a tensor"),
        case(
            "op = Pair(cols=2, src=np.zeros((2, 2), np.uint16))\nr = op.dst",
            case_="a signature tensor not given",
        ),
        case(
            (
                "op = Pair(cols=3, kind='f32')\n"
                "s, d = np.ones((2, 3), np.float32), "
                "np.zeros((2, 3), np.float32)\n"
                "o2 = op.with_tensors({'src': s})\n"
                "o3 = o2.with_tensors({'dst': d})\n"
                "r = {'types': [type(o2).__name__, "
                "type(o3).__name__], 'keys': [o2.key(), "
                "o3.key()], 'o2 src': o2.src is s, 'o3 "
                "given': sorted(o3.tensors), 'op given': "
                "sorted(op.tensors)}"
            ),
            case_="with_tensors makes new ops with only the tensors given",
        ),
        case(
            "class Deep(Pair, name='decl.pair_deep'):\n"
            "    depth = Choice(1, 2, default=2)\n"
            "r = {'fields': list(Deep.fields()), 'key': Deep(cols=5).key(), "
            "'signature': sorted(Deep(cols=5).signature())}",
            case_="a subclass adds a field, keeps fields and signature",
        ),
        case(
            "class Fixed(Pair, name='decl.pair_fixed'):\n"
            "    kind = 'f32'\n"
            "r = {'fields': list(Fixed.fields()), 'key': Fixed(cols=1).key(), "
            "'bf16 refused': Fixed.fields()['kind'].options}",
            case_="a plain class attribute leaves the base's Choice a field",
        ),
        case(
            "r = type('Bad', (Spec,), {'rank': Int(default=0), 'with_tensors': Choice('a')}, "
            "name='decl.bad_fields')",
            case_="fields named as an op's own attributes",
        ),
        case(
            "r = type('Worse', (Spec,), {'tensors': Int(default=1)})",
            case_="the shadow check comes before the missing name",
        ),
    ]


def group_ops():
    """Ops of a group spec with handles and shared (group) tensors."""
    rows = [
        ("no handle (as check makes it)", "Gather(size=3, part=2)"),
        ("rank 2's handle", "Gather(size=3, part=2, group=crew[2])"),
        (
            "a handle of a group of 2",
            "Gather(size=3, part=2, group=pair[1])",
        ),
        ("a size below the field's lo", "Gather(size=1, part=2)"),
        (
            "a handle for a spec that is no group spec",
            "Pair(cols=2, group=crew[0])",
        ),
        (
            "a shared tensor in a spec that is no group spec",
            "Lonely(m=4)",
        ),
        (
            "a plain array for the shared tensor, no handle",
            "Gather(size=3, part=2, whole=np.zeros((3, 2), np.float32))",
        ),
        (
            "the group's copy, no handle",
            "Gather(size=3, part=2, whole=whole[1])",
        ),
        (
            "a plain array with the handle",
            "Gather(size=3, part=2, group=crew[0], whole=np.zeros((3, 2), np.float32))",
        ),
        (
            "the rank's own copy with its handle",
            "Gather(size=3, part=2, group=crew[0], whole=whole[0])",
        ),
        (
            "another rank's copy with the handle",
            "Gather(size=3, part=2, group=crew[0], whole=whole[2])",
        ),
        (
            "another group's copy of the same name",
            "Gather(size=3, part=2, group=crew[0], whole=foreign)",
        ),
        (
            "the unshared input given with the handle",
            "Gather(size=3, part=2, group=crew[1], piece=np.zeros(2, np.float32))",
        ),
    ]
    return [case(f"r = placed({expr})", case_=desc) for desc, expr in rows]


def document():
    """The authored spec.json, outcomes not yet recorded."""
    return {
        "notation": NOTATION,
        "file": FILE,
        "setup": SETUP,
        "names": names(),
        "declarations": declarations(),
        "ops": ops(),
        "tensors": {"made": made(), "storage": storage()},
        "given": given(),
        "misc": misc(),
        "group_ops": group_ops(),
    }
