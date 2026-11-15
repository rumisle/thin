"""Fresh API exercises for emulated groups, schedules and shared memory."""

from __future__ import annotations

from tools.golden.opspec import case

FILE = "<test>/opspec/group.py"

NOTATION = """All constructors, schedule values, tensor identities, allocation reuse and
refusals below come from thin's real emulated group implementation.  Buffers are
assigned stable encounter labels solely to make identity visible in JSON.  The
last section stops at group.emulate's contract checks with lightweight call
records; Bound.emulate's rank-alone refusal uses the real runtime Bound.  Cases
share one namespace and the memory steps deliberately share and mutate group
state in document order."""

SETUP = """\
class Kernel:
    def __init__(self, label, arguments=None, address=None):
        self.name = label
        self.args = dict(arguments or {})
        self.address = address or ('group-address:' + label)

class GroupProbe:
    def __init__(self, kernel, handle, args=None, *, co_resident=False, grid=(1, 1, 1)):
        self.kernel, self.group = kernel, handle
        self.args = dict(args or {})
        self.grid, self.co_resident, self.overread = tuple(grid), co_resident, {}

def probe(kernel, handle, args=None, **kw):
    return GroupProbe(kernel, handle, args, **kw)

ranks = group.emulated(2)
team = ranks[0].team
labels = {}
objects = []

def label_of(value):
    key = id(value)
    if key not in labels:
        labels[key] = f'object{len(labels)}'
        objects.append(value)
    return labels[key]

def tensor_fact(rank, name, dtype, shape):
    value = ranks[rank].tensor(name, dtype, shape)
    return {'identity': label_of(value), 'storage_dtype': value.dtype.name,
            'storage_shape': list(value.shape),
            'zero_filled': bool(np.all(value.reshape(-1).view(np.uint8) == 0))}

def peer_identities(value):
    copies = team.peers_of(value)
    return None if copies is None else [label_of(c) for c in copies]

first_kernel = Kernel('alpha', address='address-A')
other_kernel = Kernel('beta', address='address-B')

def allocation_fact(kernel, name, size):
    buffers = team.mapped(kernel, name, size)
    return {'identities': [label_of(b) for b in buffers],
            'sizes': [len(b) for b in buffers], 'zero_filled': all(not any(b) for b in buffers)}

left_group, right_group = group.emulated(2), group.emulated(2)
mapped_kernel = Kernel('mapped_call', {'window': {'owned': 'mapped', 'ranks': 2, 'kind': 'ptr'}})
plain_kernel = Kernel('plain_call')
"""


def emulated():
    """Valid group handles and invalid world declarations."""
    plans = [
        (1, None),
        (3, None),
        (6, None),
        (0, None),
        (-3, None),
        (True, None),
        ("3", None),
        (3.0, None),
    ]
    out = []
    for world, _ in plans:
        text = f"handles = group.emulated({world!r})\nr = [repr(h) for h in handles] + [handles[0].device]"  # noqa: E501
        out.append(case(text, world=world, case_="positive worlds or exact world validation"))
    return out


def slow():
    """Skewed last-rank schedules, with and without residency."""
    return [
        case(
            f"r = group.slow({world}, resident={resident})",
            world=world,
            resident=resident,
        )
        if resident is not None
        else case(f"r = group.slow({world})", world=world)
        for world, resident in ((2, None), (5, None), (5, 2), (1, None))
    ]


def memory():
    """A stateful sequence over real group tensors and mapped buffers."""
    plans = [
        (
            "r = tensor_fact(0, 'matrix', 'f16', (3, 7))",
            "first rank creates all zeroed copies",
        ),
        (
            "r = tensor_fact(0, 'matrix', 'f16', (3, 7))",
            "a repeated request reuses the copy",
        ),
        (
            "r = tensor_fact(1, 'matrix', 'f16', (3, 7))",
            "another rank receives its distinct copy",
        ),
        (
            "r = tensor_fact(1, 'matrix', 'f16', [3, 7])",
            "a list shape identifies the same tensor",
        ),
        (
            "r = tensor_fact(1, 'matrix', 'f16', (6, 7))",
            "an existing name refuses another shape",
        ),
        (
            "r = tensor_fact(1, 'matrix', 'bf16', (3, 7))",
            "an existing name refuses another dtype",
        ),
        (
            "r = tensor_fact(0, 'packed', 'e2m1', (3, 8))",
            "sub-byte elements use packed zeroed storage",
        ),
        (
            "r = tensor_fact(0, 'bad', 'f8', (3,))",
            "unknown group tensor dtype is refused",
        ),
        (
            "matrix0 = ranks[0].tensor('matrix', 'f16', (3, 7))\nr = peer_identities(matrix0)",
            "every peer copy is returned for rank zero",
        ),
        (
            "r = peer_identities(ranks[1].tensor('matrix', 'f16', (3, 7)))",
            "the same list is found from rank one",
        ),
        (
            "r = peer_identities(matrix0[:1])",
            "a partial numpy view is not itself a group copy",
        ),
        (
            "r = peer_identities(matrix0.copy())",
            "equal contents do not imply group-copy identity",
        ),
        (
            "r = team.holds(first_kernel, 'window', 48)",
            "no allocation exists initially",
        ),
        (
            "r = allocation_fact(first_kernel, 'window', 48)",
            "the first allocation creates one zeroed buffer per rank",
        ),
        (
            "r = team.holds(first_kernel, 'window', 48)",
            "the exact size is held",
        ),
        (
            "r = team.holds(first_kernel, 'window', 49)",
            "a larger requirement is not held",
        ),
        (
            "r = allocation_fact(first_kernel, 'window', 24)",
            "a smaller request reuses the adequate newest allocation",
        ),
        (
            "r = allocation_fact(first_kernel, 'window', 96)",
            "a larger request allocates new zeroed buffers",
        ),
        (
            "r = allocation_fact(first_kernel, 'window', 48)",
            "later requests use the newest adequate allocation",
        ),
        (
            "r = allocation_fact(first_kernel, 'scratch', 48)",
            "another argument has an independent allocation history",
        ),
        (
            "r = allocation_fact(other_kernel, 'window', 48)",
            "another kernel address has independent buffers",
        ),
        (
            "r = allocation_fact(Kernel('alpha_alias', address='address-A'), 'window', 48)",
            "kernel address, rather than object or display name, keys allocations",
        ),
        (
            "ranks[1].close()\nr = team.holds(first_kernel, 'window', 48)",
            "closing through either rank clears shared allocations",
        ),
        (
            "r = tensor_fact(0, 'matrix', 'f16', (3, 7))",
            "a tensor request after close creates a fresh generation",
        ),
    ]
    return [case(text, step=behavior) for text, behavior in plans]


def emulate_refusals():
    """Contract failures detected before native joint emulation."""
    return [
        case(
            (
                "r = group.emulate([probe(plain_kernel, "
                "left_group[0]), probe(plain_kernel, "
                "right_group[1])])"
            ),
            case_="calls from two teams cannot be one emulated group run",
        ),
        case(
            "r = group.emulate([probe(plain_kernel, None), probe(plain_kernel, left_group[1])])",
            case_="a rank-less call cannot participate",
        ),
        case(
            "r = group.emulate([probe(mapped_kernel, left_group[0], {'window': 48}), "
            "probe(mapped_kernel, left_group[1], {'window': 96})])",
            case_="ranks must bind a mapped argument to the same byte count",
        ),
        case(
            "r = group.emulate([probe(plain_kernel, left_group[0]), "
            "probe(mapped_kernel, left_group[0], {'window': 48}), "
            "probe(mapped_kernel, left_group[1], {'window': 96})])",
            case_="the mapped-size check covers kernels after the first call",
        ),
        case(
            (
                "r = group.emulate([probe(plain_kernel, "
                "left_group[0], co_resident=True, grid=(3, "
                "1, 1)), probe(plain_kernel, left_group[1], "
                "co_resident=True, grid=(3, 1, 1))], "
                "{'resident': 2})"
            ),
            case_="a direct group emulation refuses an impossible co-residency schedule",
        ),
        case(
            "r = Bound(plain_kernel, (1, 1, 1), {}, group=left_group[0]).emulate()",
            case_="Bound.emulate refuses to run only one rank of a group call",
        ),
    ]


def document():
    """The authored group.json document, with outcomes left to thin."""
    return {
        "notation": NOTATION,
        "file": FILE,
        "setup": SETUP,
        "emulated": emulated(),
        "slow": slow(),
        "memory": {"steps": memory()},
        "emulate_refusals": emulate_refusals(),
    }
