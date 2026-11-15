"""New executable checks of emulated/GPU call scheduling and group checking."""

from __future__ import annotations

from tools.golden.opspec import case

FILE = "<test>/opspec/runner.py"

NOTATION = """Each call is executed by thin.opspec.check, with host arithmetic confined to
small Probe calls returned by real registered bindings.  Probe.emulate/run logs
the call and writes or faults; it does not implement any check/selection logic.
A result is the list of Failures expanded to every public scenario field plus
rerun() and str().  LOG is recorded separately after every case, including
refusals.  The gpu case stubs only tensor transfer and synchronization in the
replay harness, exactly as the public replay test does.  The group section uses
real emulated group handles and tensors, replacing only group.emulate with a
recording dispatcher that computes the prescribed rank-wise result."""

SETUP = """\
class Kernel:
    def __init__(self, label):
        self.name, self.args, self.address = label, {}, 'runner:' + label

LOG = []

class Probe:
    def __init__(self, kernel, args, *, grid=(3, 1, 1), co_resident=False,
                 action=None, fault=None):
        self.kernel, self.args, self.grid = kernel, dict(args), tuple(grid)
        self.co_resident, self.overread, self.group = co_resident, {}, None
        self.action, self.fault = action, fault
    def _invoke(self, sched=None, method='emulate'):
        entry = {'call': method, 'kernel': self.kernel.name}
        if method == 'emulate':
            entry['sched'] = sched
        LOG.append(entry)
        if self.fault is not None:
            raise DeviceFault(self.fault)
        if self.action is not None:
            self.action(self)
    def emulate(self, sched=None):
        self._invoke(sched)
    def run(self):
        self._invoke(method='run')

class Accumulate(Spec, name='authored.accumulate'):
    length = Int(lo=2)
    dists = {'amplitude': [1.0, 1.0e20]}
    def signature(op):
        return {'left': In('f32', op.length), 'right': In('f32', op.length),
                'answer': Out('f32', op.length)}
    def inputs(op, seed, *, amplitude=1.0):
        positions = np.arange(op.length, dtype=np.float64)
        left = ((2 * positions + seed + 1) * amplitude).astype(np.float32)
        right = np.full(op.length, (seed + 3) * amplitude, np.float64).astype(np.float32)
        return {'left': left, 'right': right}
    def check(op, inputs, outputs):
        want = inputs['left'] + inputs['right']
        got = outputs['answer']
        wrong = np.flatnonzero(got.view(np.uint32) != want.view(np.uint32))
        if not len(wrong):
            return None
        at = int(wrong[0])
        return f'answer[{at}] = {float(got[at])}, want {float(want[at])} ({len(wrong)} wrong)'

def addition_action(omit_last=False):
    def invoke(call):
        value = call.args['left'] + call.args['right']
        target = call.args['answer']
        if omit_last:
            target[:-1] = value[:-1]
        else:
            target[:] = value
    return invoke

def register(label, *, tunables=None, omit_last=False, fault=None, co_resident=False, calls=1):
    @implements(Accumulate, tunables=tunables,
                needs={'co_resident': True} if co_resident else None)
    def bind(op, **chosen):
        return [Probe(Kernel(f'{label}#{index}'),
                      {'left': op.left, 'right': op.right, 'answer': op.answer},
                      co_resident=co_resident, action=addition_action(omit_last), fault=fault)
                for index in range(calls)]
    bind.__name__ = bind.__qualname__ = label
    bind.implementation.fn.__name__ = label
    return bind

register('accurate', tunables={'lanes': [16, 32]})
register('truncated', omit_last=True, tunables={'lanes': [16]})
register('faulting', fault='synthetic missing memory')
register('resident', co_resident=True)
register('pipeline', calls=2)

def failure_facts(op, **arguments):
    LOG.clear()
    failures = check(op, **arguments)
    return [{'impl': f.impl, 'tunables': dict(f.tunables), 'seed': f.seed,
             'dist': dict(f.dist), 'sched': f.sched, 'device': f.device,
             'why': f.why, 'rerun': f.rerun(), 'str': str(f)} for f in failures]
"""


def runner_cases():
    """The single-rank check scheduling and Failure matrix."""
    rows = [
        (
            "r = failure_facts(Accumulate(length=7), device='emulated')",
            "three seeds, two distribution points and all default schedules",
        ),
        (
            "r = failure_facts(Accumulate(length=7), device='emulated', seeds=(4,), "
            "scheds=({'order': 'reverse'},))",
            "one seed and one schedule still visit both distribution points",
        ),
        (
            "r = failure_facts(Accumulate(length=7), impl='accurate', lanes=32, "
            "seeds=(3,), scheds=(None,))",
            "an explicit implementation and tunable are carried through checking",
        ),
        (
            "r = failure_facts(Accumulate(length=7), impl='truncated', seeds=(2,), "
            "scheds=(None, {'jitter': 2}))",
            "a wrong output retains implementation, tunable, seed, dist, schedule and rerun text",
        ),
        (
            "r = failure_facts(Accumulate(length=7), impl='truncated', seeds=(1,), "
            "scheds=(None,), dists={'amplitude': [3.0]})",
            "a distribution override replaces the declared points",
        ),
        (
            "r = failure_facts(Accumulate(length=7), seeds=(0,), scheds=(None,), dists={})",
            "an empty override uses inputs defaults and yields one point",
        ),
        (
            "r = failure_facts(Accumulate(length=7), seeds=(0,), scheds=(None,), "
            "dists={'amplitude': []})",
            "a named distribution with no values is refused",
            False,
        ),
        (
            "r = failure_facts(Accumulate(length=7), dists={'offset': [2]})",
            "an unknown distribution key is refused before running",
            False,
        ),
        (
            "r = failure_facts(Accumulate(length=7), impl='faulting', seeds=(5,), scheds=(None,))",
            "DeviceFault becomes a Failure at every distribution point",
        ),
        (
            "r = failure_facts(Accumulate(length=7), impl='resident', seeds=(0,), "
            "dists={'amplitude': [1.0]})",
            "default schedules whose residency is below the grid are skipped",
        ),
        (
            "r = failure_facts(Accumulate(length=7), impl='pipeline', seeds=(0,), "
            "dists={'amplitude': [1.0]}, scheds=(None, {'resident': 3}))",
            "multiple calls execute in binding order for each schedule",
        ),
        (
            "r = failure_facts(Accumulate(length=7), seeds=(0,), scheds=())",
            "an empty schedule collection performs no emulated runs",
            False,
        ),
        (
            "r = failure_facts(Accumulate(length=7), impl='truncated', seeds=(0, 1), "
            "dists={'amplitude': [1.0]}, device='gpu')",
            "GPU checking calls run once per seed/point and records device gpu",
        ),
    ]
    out = []
    for row in rows:
        text, behavior, *_log = row
        out.append(
            case(
                text,
                case_=behavior,
                log=True,
                gpu_copies_stubbed="device='gpu'" in text,
            )
        )
    return out


GROUP_SETUP = """\
class RankTotal(Spec, name='authored.rank_total', group='world'):
    world = Int(lo=2, hi=6)
    length = Int(lo=1)
    def signature(op):
        return {'source': In('u32', op.length), 'total': Out('u32', op.length)}
    def inputs(op, seed):
        return [{'source': (np.arange(op.length, dtype=np.uint32) + 7 * rank + seed)}
                for rank in range(op.world)]
    def check(op, inputs, outputs):
        want = sum(i['source'].astype(np.uint64) for i in inputs).astype(np.uint32)
        for rank, output in enumerate(outputs):
            wrong = np.flatnonzero(output['total'] != want)
            if len(wrong):
                at = int(wrong[0])
                return (f'rank {rank}: total[{at}] = {int(output["total"][at]):#x}, '
                        f'want {int(want[at]):#x}')
        return None

class SharedTotal(Spec, name='authored.shared_total', group='world'):
    world = Int(lo=2, hi=6)
    length = Int(lo=1)
    def signature(op):
        return {'table': In('u32', op.world, op.length, shared=True),
                'answer': Out('u32', op.length)}
    def inputs(op, seed):
        return [{'table': np.full((op.world, op.length), 20 * rank + seed + 1, np.uint32)}
                for rank in range(op.world)]
    def check(op, inputs, outputs):
        for rank, output in enumerate(outputs):
            want = sum(i['table'][rank].astype(np.uint64) for i in inputs).astype(np.uint32)
            if not np.array_equal(output['answer'], want):
                return f'rank {rank}: answer {output["answer"].tolist()} != {want.tolist()}'
        return None

ROUND_ACTION = {}
CURRENT_OP = {}

def rank_calls(op, label, count=1, co_resident=False):
    return [Probe(Kernel(f'{label}#{index}'), dict(op.tensors), grid=(2, 1, 1),
                  co_resident=co_resident) for index in range(count)]

def total_action(wrong_rank=None):
    def invoke(round_calls):
        sources = [c.args['source'] for c in round_calls]
        for call in round_calls:
            call.args['total'][:] = sum(sources)
            if call.group.rank == wrong_rank:
                call.args['total'][0] = 0
    return invoke

@implements(RankTotal)
def total(op):
    ROUND_ACTION['run'] = total_action()
    return rank_calls(op, 'total')

@implements(RankTotal)
def total_wrong(op):
    ROUND_ACTION['run'] = total_action(wrong_rank=1)
    return rank_calls(op, 'total_wrong')

@implements(RankTotal)
def total_ragged(op):
    ROUND_ACTION['run'] = total_action()
    return rank_calls(op, 'total_ragged', count=1 + op.rank)

@implements(RankTotal, needs={'co_resident': True})
def total_resident(op):
    ROUND_ACTION['run'] = total_action()
    return rank_calls(op, 'total_resident', co_resident=True)

@implements(SharedTotal)
def shared_total(op):
    def invoke(round_calls):
        for call in round_calls:
            copies = call.group.team.peers_of(call.args['table'])
            call.args['answer'][:] = sum(c[call.group.rank] for c in copies)
    ROUND_ACTION['run'] = invoke
    return rank_calls(op, 'shared_total')

def recording_group_run(calls, sched=None, staged=()):
    def staged_label(item):
        for call in calls:
            for name, value in call.args.items():
                peers = call.group.team.peers_of(value)
                if peers is not None:
                    for rank, copy in enumerate(peers):
                        if copy is item:
                            return f'{CURRENT_OP["op"].key()}.{name}[rank {rank}]'
        return '<unknown>'
    LOG.append({'call': 'group.emulate',
                'bounds': [f'{c.kernel.name} rank {c.group.rank}' for c in calls],
                'sched': sched, 'staged': [staged_label(s) for s in staged]})
    ROUND_ACTION['run'](calls[:calls[0].group.world])

def group_failure_facts(op, **arguments):
    LOG.clear()
    CURRENT_OP['op'] = op
    return [{'impl': f.impl, 'tunables': dict(f.tunables), 'seed': f.seed,
             'dist': dict(f.dist), 'sched': f.sched, 'device': f.device,
             'why': f.why, 'rerun': f.rerun(), 'str': str(f)}
            for f in check(op, **arguments)]

unmodified_group_run = group.emulate
group.emulate = recording_group_run
"""


def group_cases():
    """The joint group check matrix over real emulated handles."""
    rows = [
        (
            "r = group_failure_facts(RankTotal(world=2, length=3), impl='total', seeds=(0,))",
            "default schedules plus both group-skew schedules; each round is issued twice",
        ),
        (
            "r = group_failure_facts(RankTotal(world=3, length=3), impl='total', "
            "seeds=(0, 1), scheds=(None,))",
            "explicit non-default schedules do not gain group-skew entries",
        ),
        (
            "r = group_failure_facts(RankTotal(world=2, length=3), impl='total', seeds=(0,), "
            "scheds=tuple(list(SCHEDS)))",
            "an equal copy of defaults is recognized and gains group-skew entries",
        ),
        (
            "r = group_failure_facts(RankTotal(world=2, length=3), impl='total_wrong', "
            "seeds=(0,), scheds=(None, {'jitter': 1}))",
            "a rank's wrong output yields complete Failure records at each schedule",
        ),
        (
            "r = group_failure_facts(RankTotal(world=2, length=3), impl='total_ragged', "
            "seeds=(0,), scheds=(None,))",
            "different bounds counts on different ranks are refused before a group run",
            False,
        ),
        (
            "r = group_failure_facts(RankTotal(world=2, length=3), impl='total_resident', "
            "seeds=(0,), scheds=(None, {'resident': 1}, group.slow(2, resident=1), "
            "{'resident': 2}))",
            "residency considers every rank's co-resident calls",
        ),
        (
            "r = group_failure_facts(SharedTotal(world=2, length=3), seeds=(0,), scheds=(None,))",
            "each shared input has a real group copy and all copies are staged",
        ),
        (
            "saved_inputs = RankTotal.inputs\n"
            "RankTotal.inputs = lambda op, seed: [{'source': np.zeros(op.length, np.uint32)}]\n"
            "try:\n"
            "    r = group_failure_facts(RankTotal(world=2, length=3), impl='total', "
            "seeds=(0,), scheds=(None,))\n"
            "finally:\n"
            "    RankTotal.inputs = saved_inputs",
            "inputs returning fewer ranks than the group are refused",
            False,
        ),
    ]
    out = []
    for row in rows:
        text, behavior, *_log = row
        out.append(case(text, case_=behavior, log=True))
    return out


def document():
    """The authored runner.json document, before thin computes its outcomes."""
    from thin import opspec as _opspec

    return {
        "notation": NOTATION,
        "file": FILE,
        "setup": SETUP,
        "SCHEDS": [dict(s) if isinstance(s, dict) else s for s in _opspec.SCHEDS],
        "cases": runner_cases(),
        "group": {
            "setup": GROUP_SETUP,
            "teardown": "group.emulate = unmodified_group_run",
            "cases": group_cases(),
        },
    }
