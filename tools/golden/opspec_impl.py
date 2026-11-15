"""Independently authored scenarios for implementation registration and lowering.

The programs use small recording call objects only at the binding boundary.  All
selection, range, served-view and deployment-requirement decisions are made by
thin.opspec itself and their results are recorded by the shared replay engine.
"""

from __future__ import annotations

from tools.golden.opspec import case

FILE = "<test>/opspec/implement.py"

NOTATION = """Executable in the same namespace and document order as spec.json.  Each
scenario declares a fresh Spec subclass, registers its implementations, then runs
its lowering and candidates calls.  Kernel and Probe are deliberately minimal
stand-ins at the public binding boundary; Arg records expose the kernel argument
metadata consumed by thin.  inspect_lower records the selected implementation,
chosen tunables, bound calls, and the exact field/group/tunable view seen by the
binding.  inspect_candidates records implementation order, Cartesian tunable
order, and served ranges.  No outcome is supplied by these programs: replay
executes every statement with thin and captures returned values or refusals."""

SETUP = '''\
class Arg:
    def __init__(self, lo=None, hi=None, width=4, kind="u32"):
        self.lo, self.hi, self.bytes, self.kind = lo, hi, width, kind

class Kernel:
    def __init__(self, label, arguments=(), address=None):
        self.name = label
        self.args = dict(arguments)
        self.address = address or "authored:" + label

class Probe:
    def __init__(self, kernel, args=None, grid=(1, 1, 1), co_resident=False, overread=None):
        self.kernel = kernel
        self.args = dict(args or {})
        self.grid = tuple(grid)
        self.co_resident = co_resident
        self.overread = dict(overread or {})
        self.group = None

OBSERVED = {}

def field_fact(value):
    if isinstance(value, Field):
        return {"kind": "Field", "value": int(value), "range": list(value.range)}
    return {"kind": type(value).__name__, "value": value}

def call_fact(call):
    return {"kernel": call.kernel.name, "grid": list(call.grid),
            "co_resident": call.co_resident, "overread": dict(call.overread),
            "group": None if call.group is None else repr(call.group)}

def binding(label, plans):
    """A new binding over explicit kernel/argument/binding plans."""
    def bind(op, **tunables):
        OBSERVED.clear()
        OBSERVED.update(fields={n: field_fact(v) for n, v in op.field_values().items()},
                        group=None if op.group is None else repr(op.group),
                        tunables=dict(tunables))
        calls = []
        for plan in plans:
            records = {n: Arg(*spec) for n, spec in plan.get("arguments", {}).items()}
            kernel = Kernel(plan.get("kernel", label), records)
            args = {}
            for arg, source in plan.get("bind", {}).items():
                args[arg] = op.tensors[source] if source in op.tensors else getattr(op, source)
            calls.append(Probe(kernel, args, co_resident=plan.get("co_resident", False),
                               overread=plan.get("overread")))
        return calls[0] if len(plans) == 1 and plans[0].get("single") else calls
    bind.__name__ = bind.__qualname__ = label
    return bind

class Transform(Spec, name="authored.transform"):
    count = Int(lo=1)
    tile = Int(default=2, lo=2, hi=24)
    variant = Choice("forward", "reverse", default="forward")
    def signature(op):
        return {"src": In("f32", op.count), "dst": Out("f32", op.count)}

def buffers_for(fields):
    return {"src": np.zeros(fields["count"], np.float32),
            "dst": np.zeros(fields["count"], np.float32)}

def inspect_lower(cls, fields, impl=None, tunables=None, tensors=True):
    kw = dict(fields)
    if tensors:
        kw.update(buffers_for(fields))
    impl_record, chosen, calls = lowered(cls(**kw), impl, **(tunables or {}))
    return {"impl": impl_record.name, "tunables": chosen,
            "calls": [call_fact(c) for c in calls], "observed": dict(OBSERVED)}

def inspect_candidates(cls, fields):
    return [{"impl": record.name, "tunables": chosen,
             "served": {n: list(bounds) for n, bounds in served.items()}}
            for record, chosen, served in candidates(cls(**fields))]
'''


def scenario(name, setup, implementations, cases):
    """One self-contained registration/lowering scenario."""
    return {
        "case": name,
        "setup": setup,
        "implementations": implementations,
        "cases": cases,
    }


def implementation(text, behavior):
    """A registration statement which intentionally returns no result."""
    return case(text, case_=behavior, effect=True)


def selection():
    """Cover predicates, tunable products and explicit lowering."""
    cls = "authored.selection"
    setup = f"class Select(Transform, name={cls!r}):\n    pass"
    impls = [
        implementation(
            "implements(Select, tunables={'block': [32, 96], 'unroll': [1, 3]}, "
            "covers=lambda op, **t: op.count <= 80 and t['block'] * t['unroll'] >= 96)"
            "(binding('filtered', [{'arguments': {'count': (1, 4096, 4)}, "
            "'bind': {'src': 'src', 'dst': 'dst', 'count': 'count'}}]))",
            "a predicate and two tunables in product order",
        ),
        implementation(
            "implements(Select, tunables={'block': [192]})(binding('fallback', "
            "[{'arguments': {'count': (1, 4096, 4)}, "
            "'bind': {'src': 'src', 'dst': 'dst', 'count': 'count'}}]))",
            "an unconditional fallback",
        ),
    ]
    rows = [
        ({"count": 9}, "", "first covering pair and all product pairs"),
        ({"count": 900}, "", "only the fallback covers a large count"),
        (
            {"count": 9},
            ", impl='filtered'",
            "the named pair defaults to its first, non-covering tunables",
        ),
        (
            {"count": 9},
            ", impl='filtered', tunables={'unroll': 3}",
            "one override retains defaults for the other tunables",
        ),
        (
            {"count": 9},
            ", impl='filtered', tunables={'block': 96, 'unroll': 1}",
            "both tunables explicitly selected",
        ),
        (
            {"count": 9},
            ", impl='filtered', tunables={'block': 777, 'unroll': 3}",
            "an explicitly named pair accepts an experimental tunable value",
        ),
        (
            {"count": 900},
            ", impl='filtered'",
            "the named implementation's predicate refuses the op",
        ),
        ({"count": 9}, ", impl='missing'", "an unknown short name"),
        (
            {"count": 9},
            ", tunables={'block': 32}",
            "tunables cannot be supplied without an implementation",
        ),
        (
            {"count": 9},
            ", tensors=False",
            "lowering refuses before selection when signature tensors are absent",
        ),
    ]
    cases = []
    for fields, extra, behavior in rows:
        node = case(f"r = inspect_lower(Select, {fields!r}{extra})", behavior=behavior)
        node["lowered_call"] = node.pop("call")
        node["candidates_call"] = f"r = inspect_candidates(Select, {fields!r})"
        node["fields"] = dict(fields)
        cases.append(node)
    return scenario("covering pairs and explicit tunables", setup, impls, cases)


def served_ranges():
    """Valid served intervals and the views shown to bindings."""
    setup = "class Served(Transform, name='authored.served'):\n    pass"
    impls = [
        implementation(
            "implements(Served, tunables={'width': [16]}, serves=lambda op, **_: "
            "{'count': (1, 48)})(binding('local', [{'arguments': {'count': (1, 4096, 4)}, "
            "'bind': {'count': 'count', 'src': 'src', 'dst': 'dst'}}]))",
            "a narrow served range",
        ),
        implementation(
            "implements(Served, serves=lambda op, **_: {'count': (1, 5000), 'tile': (2, 24)})"
            "(binding('wide', [{'arguments': {'count': (1, 8192, 4), 'tile': (None, None, 4)}, "
            "'bind': {'count': 'count', 'tile': 'tile', 'src': 'src', 'dst': 'dst'}}]))",
            "two served Int fields including open kernel metadata",
        ),
    ]
    rows = [
        ({"count": 12}, "", "the binding receives one Field range"),
        ({"count": 60}, "", "the fallback receives two Field ranges"),
        (
            {"count": 60},
            ", impl='local'",
            "an explicitly selected range explains why it does not cover",
        ),
    ]
    cases = []
    for fields, extra, behavior in rows:
        node = case(f"r = inspect_lower(Served, {fields!r}{extra})", behavior=behavior)
        node["lowered_call"] = node.pop("call")
        node["candidates_call"] = f"r = inspect_candidates(Served, {fields!r})"
        cases.append(node)
    return scenario("served ranges become coverage and Field values", setup, impls, cases)


def invalid_served():
    """Refusals for malformed served-field declarations."""
    setup = "class InvalidRanges(Transform, name='authored.invalid_ranges'):\n    pass"
    plans = [
        ("not_int", "{'variant': (0, 1)}", "a Choice is not a served Int field"),
        ("below", "{'tile': (1, 8)}", "the lower end precedes the field bound"),
        ("above", "{'tile': (2, 25)}", "the upper end exceeds the field bound"),
        ("reversed", "{'count': (9, 2)}", "crossed served bounds"),
        ("unknown", "{'mystery': (1, 4)}", "an unknown served field"),
    ]
    impls = [
        implementation(
            f"implements(InvalidRanges, serves=lambda op, **_: {expr})"
            f"(binding({name!r}, [{{'arguments': {{'count': (1, 4096, 4)}}}}]))",
            behavior,
        )
        for name, expr, behavior in plans
    ]
    cases = []
    for index, (name, _expr, behavior) in enumerate(plans):
        fields = {"count": 5}
        extra = "" if index == 0 else f", impl={name!r}"
        node = case(
            f"r = inspect_lower(InvalidRanges, {fields!r}{extra})",
            behavior=behavior,
        )
        node["lowered_call"] = node.pop("call")
        node["candidates_call"] = f"r = inspect_candidates(InvalidRanges, {fields!r})"
        cases.append(node)
    return scenario("invalid served declarations", setup, impls, cases)


def honoured_ranges():
    """Kernel argument intervals honoring served field intervals."""
    setup = "class Honoured(Transform, name='authored.honoured'):\n    pass"
    plans = [
        (
            "too_narrow",
            "{'count': (1, 5000)}",
            [{"arguments": {"count": (1, 512, 4)}}],
            "the served interval exceeds an explicit kernel interval",
        ),
        (
            "absent_argument",
            "{'count': (1, 48)}",
            [{"arguments": {"src": (None, None, 8)}}],
            "no bound kernel takes the served field",
        ),
        (
            "narrow_unsigned",
            "{'count': (1, 70000)}",
            [{"arguments": {"count": (None, None, 2)}}],
            "open u16 kernel bounds imply 0 through 65535",
        ),
        (
            "wide_unsigned",
            "{'count': (1, 70000)}",
            [{"arguments": {"count": (None, None, 4)}}],
            "open u32 kernel bounds honour the interval",
        ),
        (
            "second_too_narrow",
            "{'count': (1, 48)}",
            [
                {"kernel": "first", "arguments": {"count": (1, 48, 4)}},
                {"kernel": "second", "arguments": {"count": (2, 48, 4)}},
            ],
            "every bound kernel, not only the first, must honour the range",
        ),
        ("no_calls", "{'count': (1, 48)}", [], "an empty binding has no served argument"),
    ]
    impls = [
        implementation(
            f"implements(Honoured, serves=lambda op, **_: {serves})"
            f"(binding({name!r}, {bound_plans!r}))",
            behavior,
        )
        for name, serves, bound_plans, behavior in plans
    ]
    cases = []
    for name, _serves, _bound_plans, behavior in plans:
        fields = {"count": 5}
        node = case(
            f"r = inspect_lower(Honoured, {fields!r}, impl={name!r})",
            behavior=behavior,
        )
        node["lowered_call"] = node.pop("call")
        node["candidates_call"] = f"r = inspect_candidates(Honoured, {fields!r})"
        cases.append(node)
    return scenario("bound kernels must honour served ranges", setup, impls, cases)


def requirements():
    """Deployment requirements and matching bound-call properties."""
    setup = "class Requirements(Transform, name='authored.requirements'):\n    pass"
    rejected = [
        ("unknown_need", "{'spin': True}", "an unknown deployment requirement"),
        ("zero_overread", "{'overread': {'src': 0}}", "overread bytes must be positive"),
        ("overread_not_table", "{'overread': [8]}", "overread must be a tensor table"),
    ]
    impls = [
        implementation(
            f"implements(Requirements, needs={needs})(binding({name!r}, []))",
            behavior,
        )
        for name, needs, behavior in rejected
    ]
    valid = [
        (
            "co_required_not_bound",
            "needs={'co_resident': True}",
            [{}],
            "a requirement with no co-resident bound",
        ),
        (
            "co_bound_not_required",
            "needs=None",
            [{"co_resident": True}],
            "an undeclared co-resident bound",
        ),
        (
            "overread_too_large",
            "needs={'overread': {'src': 32}}",
            [{"bind": {"src": "src"}, "overread": {"src": 96}}],
            "a bound reads farther than declared",
        ),
        (
            "overread_undeclared_tensor",
            "needs=None",
            [{"bind": {"dst": "dst"}, "overread": {"dst": 4}}],
            "a bound overreads a tensor omitted from needs",
        ),
        (
            "overread_unknown_tensor",
            "needs={'overread': {'side': 4}}",
            [{}],
            "needs names a tensor outside the signature",
        ),
        (
            "overread_non_tensor_arg",
            "needs={'overread': {'src': 32}}",
            [
                {
                    "arguments": {"count": (1, 4096, 4)},
                    "bind": {"count": "count"},
                    "overread": {"count": 4},
                }
            ],
            "an argument not holding an op tensor cannot satisfy overread needs",
        ),
        (
            "compatible",
            "needs={'co_resident': True, 'overread': {'src': 32}}",
            [
                {
                    "kernel": "lead",
                    "bind": {"src": "src"},
                    "co_resident": True,
                    "overread": {"src": 32},
                },
                {"kernel": "tail", "bind": {"src": "src"}, "overread": {"src": 8}},
            ],
            "co-residency and both overread limits agree",
        ),
        ("single_probe", "needs=None", [{"single": True}], "one Probe may be returned directly"),
    ]
    for name, need_kw, bound_plans, behavior in valid:
        text = (
            f"implements(Requirements, {need_kw})(binding({name!r}, {bound_plans!r}))"
            if need_kw != "needs=None"
            else f"implements(Requirements)(binding({name!r}, {bound_plans!r}))"
        )
        impls.append(implementation(text, behavior))
    cases = []
    for name, _need_kw, _bound_plans, behavior in valid:
        fields = {"count": 5}
        node = case(
            f"r = inspect_lower(Requirements, {fields!r}, impl={name!r})",
            behavior=behavior,
        )
        node["lowered_call"] = node.pop("call")
        node["candidates_call"] = f"r = inspect_candidates(Requirements, {fields!r})"
        cases.append(node)
    return scenario("declared and bound deployment needs", setup, impls, cases)


def no_implementation():
    """Selection and candidates on a class with no implementations."""
    setup = "class Empty(Transform, name='authored.empty'):\n    pass"
    cases = []
    for extra, behavior in (
        ("", "no implementation covers the op"),
        (", impl='anything'", "no registered name exists"),
    ):
        fields = {"count": 5}
        node = case(f"r = inspect_lower(Empty, {fields!r}{extra})", behavior=behavior)
        node["lowered_call"] = node.pop("call")
        node["candidates_call"] = f"r = inspect_candidates(Empty, {fields!r})"
        cases.append(node)
    return scenario("a spec without implementations", setup, [], cases)


OTHER_SETUP = """\
class Parent(Transform, name='authored.parent'):
    pass
implements(Parent)(binding('parent_impl', [{'bind': {'src': 'src', 'dst': 'dst'}}]))
class Child(Parent, name='authored.child'):
    pass
class Inventory(Transform, name='authored.inventory'):
    pass
array4 = np.zeros(4, np.float32)

class TeamOp(Spec, name='authored.team', group='world'):
    world = Int(lo=2, hi=6)
    count = Int(lo=1)
    def signature(op):
        return {'src': In('f32', op.count)}
implements(TeamOp, tunables={'mode': [1]})(binding('team_plain', [{'bind': {'src': 'src'}}]))
class TeamServed(TeamOp, name='authored.team_served'):
    pass
implements(TeamServed, serves=lambda op, **_: {'count': (1, 48)})(
    binding('team_served', [{'arguments': {'count': (1, 48, 4)}, 'bind': {'count': 'count'}}]))

class SharedTeam(Spec, name='authored.shared_team', group='world'):
    world = Int(lo=2, hi=6)
    count = Int(lo=1)
    def signature(op):
        return {'shared': In('u32', op.count, shared=True)}
implements(SharedTeam, serves=lambda op, **_: {'count': (1, 48)})(binding('shared_served', []))

team_handles = group.emulated(2)
shared_copy = team_handles[0].tensor('shared_words', 'u32', (4,))

def inspect_team(cls):
    OBSERVED.clear()
    tensors = {'src': np.zeros(4, np.float32)} if 'src' in cls(count=4, world=2).signature() else {}
    op = cls(world=2, count=4, group=team_handles[1], **tensors)
    record, chosen, calls = lowered(op)
    return {'impl': record.name, 'calls': [call_fact(c) for c in calls],
            'observed': dict(OBSERVED)}

class NoValues(Transform, name='authored.no_values'):
    pass
"""


def other():
    """Per-class registries and group-aware lowering outside the matrix."""
    return [
        case(
            "r = inspect_lower(Child, {'count': 4})",
            case_="implementation registries are per class, not inherited",
        ),
        case(
            "r = inspect_lower(Child, {'count': 4}, impl=implementations(Parent)[0].fn)",
            case_="a registered function still must belong to this exact spec",
        ),
        case(
            "r = lowered(Parent(count=4, src=array4, dst=array4), "
            "implementations(Parent)[0].fn)[0].name",
            case_="the registered binding function itself is a selector",
        ),
        case(
            "r = [record.name for record in implementations(Inventory)]",
            case_="a fresh class has an empty declaration-order inventory",
        ),
        case(
            "r = inspect_team(TeamOp)",
            case_="lowering assigns the rank handle to every returned call",
        ),
        case(
            "r = lower(TeamOp(world=2, count=4, src=array4))",
            case_="a group op cannot lower as a rank-less call",
        ),
        case(
            "r = inspect_team(TeamServed)",
            case_="the served-field view preserves the rank handle visible to the binding",
        ),
        case(
            "r = lower(SharedTeam(world=2, count=4, shared=shared_copy, group=team_handles[0]))",
            case_="a shared group tensor survives view reconstruction before argument validation",
        ),
        case(
            "r = lower(Parent(count=4, src=array4, dst=array4), impl=lambda op: [])",
            case_="an arbitrary callable is not an implementation selector",
        ),
        case(
            "r = implements(NoValues, tunables={'block': ()})",
            case_="empty tunable domains are refused at declaration",
        ),
        case(
            "r = lower(Parent(count=4, src=array4, dst=array4), impl='parent_impl', mystery=1)",
            case_="an explicitly selected implementation refuses undeclared tunables",
        ),
    ]


def document():
    """The complete authored implement.json program, without recorded outcomes."""
    return {
        "notation": NOTATION,
        "file": FILE,
        "setup": SETUP,
        "scenarios": [
            selection(),
            served_ranges(),
            invalid_served(),
            honoured_ranges(),
            requirements(),
            no_implementation(),
        ],
        "other_setup": OTHER_SETUP,
        "other": other(),
    }
