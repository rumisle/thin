"""The Builder: the native Builder behind the author's Python surface."""

from __future__ import annotations

import contextlib
import copy

from thin import _native
from thin.codegen import current
from thin.dsl._args import Entry
from thin.dsl._expr import Tree, leaf
from thin.dsl._lds import Layout
from thin.dsl._maker import _LIT, OWNER, Counter, _mfma_opts, native_reg, operand
from thin.dsl._values import BUILDERS, RegFile, TypedValue, as_reg, capture, typed
from thin.ir import B32, LOCS, ExecRegion, Instr, Loop, Program, Value, VReg, _Row
from thin.target import PHYS, Phys, target


class PathState:
    """A path state: the native one's slot and the author's records."""

    __slots__ = ("_b", "slot")

    def __init__(self, b, slot):
        self._b, self.slot = b, slot


def _cond(c) -> tuple:
    if isinstance(c, Tree):
        return ("t", c._n), c._n.text()
    if isinstance(c, Instr):
        return ("i", c._row.ins), repr(c)
    return ("o", repr(c)), repr(c)


def _lane_cond(c) -> tuple:
    if c is None:
        return ("n",), "None"
    if isinstance(c, Tree):
        return ("t", c._n), c._n.text()
    if isinstance(c, Instr):
        return ("i", c._row.ins), repr(c)
    if isinstance(c, TypedValue):
        return ("m", c._typed), repr(c)
    if isinstance(c, bool):
        return ("s", repr(c)), repr(c)
    if isinstance(c, int):
        return ("int", c), repr(c)
    if isinstance(c, Phys):
        return ("s", c.name), c.name
    return ("s", str(c)), repr(c)


def _save(save) -> tuple:
    if save is None or save is False:
        return ("n",)
    if save is True:
        return ("fresh",)
    r = as_reg(save)
    return ("r", native_reg(r), repr(r))


def _arg(x):
    if x is None:
        return None
    if isinstance(x, Tree):
        from thin.ir import FlowError

        raise FlowError(
            f"loop: {x.text}: a count or a start is an int or a scalar u32 value "
            f"(an expression is written into one first: S.u32({x.text}))"
        )
    if isinstance(x, (TypedValue, VReg)):
        return ("r", native_reg(as_reg(x)), 0)
    if isinstance(x, bool):
        return ("b", x)
    if isinstance(x, int):
        return ("i", x)
    return ("s", str(x))


def _thing(x) -> tuple:
    wait = getattr(x, "__asm_wait__", None)
    if wait is not None and not isinstance(x, type):
        return x.__asm_wait__(None)
    if isinstance(x, Counter):
        return ("c", str(x))
    if isinstance(x, str):
        return ("s", x)
    if isinstance(x, (list, tuple)):
        return ("l", [_thing(e) for e in x])
    r = as_reg(x)
    if isinstance(r, VReg):
        return ("r", native_reg(r))
    return ("o", repr(x))


class LoopHandle:
    """A loop's handle: its index and its breaks."""

    __slots__ = ("_b",)

    def __init__(self, b):
        self._b = b

    @property
    def i(self):
        """A counted loop's index."""
        n = _native.loop_index(self._b._n)
        return None if n is None else typed(n)

    def break_when(self, c, comment=None) -> None:
        """Leaves the loop when c holds."""
        b = self._b
        b._at()
        native, display = _cond(c)
        _native.loop_break(b._n, native, display, comment)


class Builder:
    """The kernel being built."""

    def __init__(self, *, vgprs=None, sgprs=None):
        """An empty kernel for the current target; a size not given is the target's default."""
        from thin.codegen import take_placement

        self.target = target()
        placement = take_placement(self)
        self._n = _native.DslBuilder(self.target.name, vgprs, sgprs, placement)
        BUILDERS[self._n.id] = self
        self._program = Program(self.target)
        self.v, self.s, self.a = RegFile(self, "v"), RegFile(self, "s"), RegFile(self, "a")
        self.files = {"v": self.v, "s": self.s, "a": self.a}
        self.ablate: set = set()
        """The ablation classes the build drops: a live set."""
        self._records: dict = {0: {}}
        self._ids: list[int] = []
        self._index: dict[int, int] = {}
        self._options = current()
        self._n.set_options(self._options.audit, self._options.safe_barrier)
        self.layout = None
        self.entered = None

    # --- Syncing the native kernel ---------------------------------------

    def _at(self) -> None:
        """Hands the native Builder this operation's location and the ablation set now.

        The set is live: its contents at each operation decide.
        """
        loc, where = capture()
        self._n.at(loc, where)
        self._n.set_ablate(sorted(self.ablate))

    def _sync_values(self, links=False) -> list:
        """The values, the new ones made; ``links`` re-reads every pin and same_as."""
        values = self._program.values
        new = _native.values(self._n, len(values))
        every = _native.value_links(self._n) if links else ()
        for file, n, align, name, loc, index, pin, same in new:
            v = Value(len(values), file, n, align, B32, name, loc, index)
            v.pin, v.same_as = pin, None if same is None else values[same]
            OWNER[v] = self._n.id
            values.append(v)
        for v, (pin, same) in zip(values, every, strict=False):
            v.pin = pin
            v.same_as = None if same is None else values[same]
        return values

    def _sync(self):
        values = self._sync_values()
        start, rows = _native.sync(self._n)
        body = self._program.body
        tgt = self.target
        fresh = []
        for gone in self._ids[start:]:
            del self._index[gone]
        del self._ids[start:]
        for row_id, ins, loc, regs, hazard in rows:
            self._index[row_id] = len(self._ids)
            self._ids.append(row_id)
            fid, name = ins.form()
            desc = tgt._forms[fid] if fid >= 0 else tgt._own[name]
            vregs = tuple(
                None if r is None else VReg(_LIT if r[0] < 0 else values[r[0]], r[1], r[2])
                for r in regs
            )
            loc = LOCS.name("hazard", loc) if hazard else loc
            fresh.append(_Row(desc, ins, vregs, loc, body))
        changed = bool(fresh) or start < len(body._layout)
        body._layout[start:] = fresh
        if changed:
            self._refresh_records()
        return body

    def _row(self, row_id):
        if row_id is None:
            return None
        self._sync()
        return self._at_row(row_id)

    @property
    def program(self) -> Program:
        """The kernel's IR: its stream, values, loops and lane regions (synced)."""
        self._sync()
        return self._program

    @property
    def ins(self):
        """The stream."""
        return self._sync()

    @property
    def values(self) -> list:
        """The program's values."""
        return self._sync_values()

    # --- State ------------------------------------------------------------

    @property
    def options(self):
        """The build options."""
        return self._options

    @options.setter
    def options(self, o) -> None:
        self._options = o
        self._n.set_options(o.audit, o.safe_barrier)

    @property
    def pass_times(self) -> dict:
        """Seconds per pass name, in the order first recorded."""
        return dict(_native.pass_times(self._n))

    @property
    def final_report(self) -> list[str]:
        """The render's LDS proof report and the ablation line."""
        return _native.final_report(self._n)

    @property
    def sealed(self) -> bool:
        """Whether the passes have run."""
        return self._n.sealed

    @property
    def hazards(self):
        """The hazard instructions the pipeline placed, or None."""
        return self._n.hazards

    @hazards.setter
    def hazards(self, n) -> None:
        self._n.hazards = n

    @property
    def waits_placed(self) -> bool:
        """Whether the waits pass has run."""
        return self._n.waits_placed

    @waits_placed.setter
    def waits_placed(self, v) -> None:
        self._n.waits_placed = v

    @property
    def needs(self) -> dict:
        """The kernel's needs."""
        co, same = self._n.needs()
        out: dict = {} if co is None else {"co_resident": co}
        if same:
            out["same_grid"] = dict(same)
        return out

    @property
    def wave_reg(self):
        """The SGPR holding the wave's index, or None."""
        r = self._n.wave_info()[0]
        return None if r is None else VReg(self.values[r[0]], r[2], r[3])

    @property
    def nwaves(self) -> int:
        """The workgroup's waves (0 before ``wave_index``)."""
        return self._n.wave_info()[1]

    @property
    def by_wave(self) -> bool:
        """Whether the passes treat each wave class apart."""
        return self._n.wave_info()[2]

    def wave_index(self, reg, nwaves, by_wave=False) -> None:
        """Names the SGPR holding the wave's index."""
        self._n.wave_index(native_reg(as_reg(reg)), nwaves, by_wave)

    @property
    def deferred(self) -> list:
        """The records for the pipeline."""
        self._sync()
        return [_deferred(self, d) for d in _native.records(self._n)[2]]

    def _refresh_records(self) -> None:
        loops, regions, _ = _native.records(self._n)
        values = self._program.values
        self._program.loops = [
            Loop(s, self._at_row(f), self._at_row(la), c) for s, f, la, c in loops
        ]
        self._program.regions = [
            ExecRegion(k, self._at_row(e), None if s is None else VReg(values[s[0]], s[2], s[3]), c)
            for k, e, s, c in regions
        ]

    def _at_row(self, row_id):
        """The instruction of a native row, or None."""
        k = self._index.get(row_id)
        return None if k is None else Instr._of(self._program.body._layout[k])

    # --- Positions and lines ---------------------------------------------

    def seal(self) -> None:
        """Settles, then refuses every later emission."""
        self._n.seal()

    def settle(self) -> None:
        """Refuses a loop left by break, then closes the open chain."""
        self._n.settle()

    @property
    def pos(self) -> int:
        """The stream length minus the hazard instructions."""
        return self._n.pos()

    @property
    def last(self):
        """The stream's last instruction, or None."""
        self._n.settle()
        body = self.ins
        return body[-1] if len(body) else None

    def emitted(self, start=0, stop=None):
        """The instructions between two positions, hazard ones included."""
        a, z = self._n.emitted(start, stop)
        body = self.ins
        return body if start == 0 and stop is None else body[a:z]

    def issued(self, counter) -> int:
        """The ops of that counter issued on the current path."""
        return self._n.issued(counter)

    def overlap(self, r1, r2) -> bool:
        """Same file and overlapping numbers."""
        a, b = as_reg(r1), as_reg(r2)
        if a.file != b.file:
            return False
        f = self.files[a.file]
        x, y = f.number(a), f.number(b)
        return x < y + b.n and y < x + a.n

    def comment(self, text) -> None:
        """Pushes a comment line."""
        self._n.comment(text)

    def put_label(self, name, from_branches=False) -> None:
        """Places a label."""
        self._n.put_label(name, from_branches)

    def new_label(self, stem) -> str:
        """``.L<stem>_<k>``."""
        return self._n.new_label(stem)

    def emit(self, x, abl=None):
        """Emits an instruction, applying host hooks and capturing its location."""
        emit = getattr(type(x), "__asm_emit__", None)
        if emit is not None:
            return emit(x, self, abl or ())
        if not isinstance(x, Instr):
            raise TypeError(f"emit({x!r}): an instruction")
        loc = x.loc or capture()[0]
        self._at()
        return self._row(self._n.emit(x._row.ins, sorted(set(abl or ())), loc))

    # --- Claims and statements -------------------------------------------

    def discard(self, reg, why) -> None:
        """Claims the register's contents are not read."""
        self._n.discard(_claim(reg), why)

    def uniform(self, *regs, why) -> None:
        """Claims these SGPRs are alike in every wave."""
        self._n.uniform([_claim(r) for r in regs], why)

    def undefined(self, *regs, why, edge=None) -> None:
        """Claims these registers hold nothing a read uses.

        From here on; on the fall-through into a label that follows; or, with
        ``edge="taken"``, on the next branch's taken edge.
        """
        self._n.undefined([_claim(r) for r in regs], why, edge)

    def assume(self, x, *, lo=0, hi, why) -> None:
        """Claims each dword is in lo..hi."""
        self._n.assume(_claim(x), repr(x), lo, hi, why)

    def nop(self, n=0) -> None:
        """Emits ``s.nop n``."""
        self._at()
        self._n.nop(n)

    def barrier(self, comment="", abl=(), fence=False) -> None:
        """Emits the target's barrier sequence."""
        self._at()
        self._n.barrier(comment, list(abl), fence)

    def need(self, what, why, arg=None) -> None:
        """Records what the binding must honour."""
        self._n.need(what, why, arg)

    @contextlib.contextmanager
    def ablatable(self, *classes):
        """Opens ablation classes for the block."""
        self._n.open_ablatable([c for c in classes if c is not None])
        try:
            yield
        finally:
            self._n.close_ablatable()

    def ablated(self, abl=()) -> bool:
        """Whether an instruction of these classes would be dropped."""
        self._n.set_ablate(sorted(self.ablate))
        return self._n.ablated(list(abl))

    @contextlib.contextmanager
    def alias_ok(self, *reads, over=(), why=""):
        """Reads the alias check must not report inside the block."""
        self._n.open_alias([_through(r) for r in reads], [_through(o) for o in over], why)
        try:
            yield
        finally:
            self._n.close_alias()

    def hold(self, label, *regs) -> None:
        """Marks the registers' dwords held by label."""
        self._n.hold(label, [native_reg(as_reg(r)) for r in regs])

    def release(self, label) -> None:
        """Unmarks the label's dwords."""
        self._n.release(label)

    # --- Path state --------------------------------------------------------

    def _apply_state_ops(self) -> None:
        for kind, a, bs in self._n.state_ops():
            if kind == "copy":
                self._records[bs[0]] = copy.deepcopy(self._records.get(a, {}))
                continue
            into = self._records.setdefault(a, {})
            for other in bs:
                _join(into, self._records.get(other, {}))

    @property
    def path_state(self) -> dict:
        """The current path's author records."""
        self._apply_state_ops()
        return self._records.setdefault(self._n.records(), {})

    def fork(self) -> PathState:
        """A copy of the current state, remembering the instructions around it."""
        return PathState(self, self._n.fork())

    @property
    def cnt(self) -> PathState:
        """The current path state."""
        return PathState(self, self._n.cnt())

    @cnt.setter
    def cnt(self, state) -> None:
        self._n.set_cnt(state.slot)

    # --- Control flow ------------------------------------------------------

    def _cond(self, c, comment=None):
        raise TypeError("a condition lowers through a construct (b.when, b.loop, b.exit_when)")

    @contextlib.contextmanager
    def _arm(self):
        try:
            yield
        except BaseException:
            _native.leave_arm(self._n, True)
            raise
        _native.leave_arm(self._n, False)

    @contextlib.contextmanager
    def when(self, c, name="if", comment=None, end=None):
        """Opens an if-chain's first arm."""
        self._at()
        native, display = _cond(c)
        _native.when(self._n, native, display, False, name, comment, end)
        with self._arm():
            yield

    @contextlib.contextmanager
    def unless(self, c, name="if", comment=None, end=None):
        """Opens an if-chain's first arm, taken when c is false."""
        self._at()
        native, display = _cond(c)
        _native.when(self._n, native, display, True, name, comment, end)
        with self._arm():
            yield

    @contextlib.contextmanager
    def elif_(self, c, comment=None):
        """Opens an elif arm."""
        self._at()
        native, display = _cond(c)
        _native.elif_(self._n, native, display, comment)
        with self._arm():
            yield

    @contextlib.contextmanager
    def otherwise(self):
        """Opens the else arm."""
        self._at()
        _native.elif_(self._n, None, "", None)
        with self._arm():
            yield

    def exit_when(self, c, to="L_end", comment=None) -> None:
        """Branches to ``to`` when c holds."""
        self._at()
        native, display = _cond(c)
        _native.exit_when(self._n, native, display, to, comment)

    def loop(
        self,
        n=None,
        *,
        until=None,
        i=None,
        start=0,
        peel=0,
        end=None,
        name=None,
        conservative=False,
    ):
        """A loop whose body runs once per copy the expansion emits."""
        self._at()
        u = None if until is None else (leaf(until), _cond(until)[1])
        it = None if i is None else i._typed
        start_given = not (isinstance(start, int) and not isinstance(start, bool) and start == 0)
        args = (_arg(n), u, it, _arg(start), start_given, peel, end, name, conservative)
        copies = _native.loop_begin(self._n, args)
        return self._copies(copies)

    def _copies(self, copies):
        handle = LoopHandle(self)
        for k in range(copies):
            _native.loop_copy_begin(self._n, k)
            try:
                yield handle
            except GeneratorExit:
                _native.loop_broken(self._n)
                return
            _native.loop_copy_end(self._n, k)
        _native.loop_end(self._n)

    def steady_state_check(self, body_fn, stem, conservative=False) -> None:
        """Runs ``body_fn`` once per pass of the steady loop body protocol."""
        self._at()
        for k in range(_native.steady_begin(self._n, stem, conservative)):
            _native.loop_copy_begin(self._n, k)
            body_fn()
            _native.loop_copy_end(self._n, k)
        _native.steady_end(self._n)

    # --- Lane regions and wave blocks --------------------------------------

    @contextlib.contextmanager
    def lanes(self, *conds, save=None, skip=False, comment=""):
        """A lane region."""
        self._at()
        pairs = [_lane_cond(c) for c in conds]
        opened = _native.lanes_begin(
            self._n, [p[0] for p in pairs], [p[1] for p in pairs], _save(save), skip, comment
        )
        if not opened:
            yield
            return
        try:
            yield
        except BaseException:
            _native.lanes_end(self._n, True)
            raise
        self._at()
        _native.lanes_end(self._n, False)

    def narrow(self, cond, comment="") -> None:
        """Narrows EXEC in place."""
        self._at()
        native, display = _lane_cond(cond)
        _native.narrow(self._n, native, display, comment)

    @contextlib.contextmanager
    def raw_exec(self, why):
        """A block whose EXEC writes the region check does not judge."""
        _native.raw_exec_begin(self._n, why)
        try:
            yield
        finally:
            _native.raw_exec_end(self._n)

    @contextlib.contextmanager
    def waves(self, waves):
        """A wave block."""
        self._at()
        _native.waves_begin(self._n, list(waves))
        try:
            yield
        except BaseException:
            _native.waves_end(self._n, True)
            raise
        self._at()
        _native.waves_end(self._n, False)

    # --- Waits ---------------------------------------------------------------

    def wait(self, *what, left=None, tags=(), written=False, comment="", abl=()) -> None:
        """Pushes the markers of a wait request."""
        self._at()
        things = [_thing(x) for x in what]
        _native.wait(self._n, things, left, [tuple(t) for t in tags], written, comment, list(abl))

    def wait_on_use(self, regs, comment="") -> None:
        """Waits for each register at its first read, for whatever load writes it."""
        regs = regs if isinstance(regs, (list, tuple)) else [regs]
        self._n.wait_on_use([native_reg(as_reg(r)) for r in regs], comment)

    def land_arguments(self, why) -> None:
        """Waits for the pending argument loads."""
        self._at()
        _native.land_arguments(self._n, why)

    # --- Entry and LDS ---------------------------------------------------------

    def entry(self, args=None, *, wg=1, preload=0, wgsize=None) -> Entry:
        """Makes the kernel's entry."""
        self._at()
        _native.entry(self._n, None if args is None else args._a, wg, preload, wgsize)
        self.entered = Entry(self, args)
        return self.entered

    def lds(self, budget=None) -> Layout:
        """Makes the kernel's LDS layout."""
        _native.lds_make(self._n, budget)
        self.layout = Layout(self)
        return self.layout

    # --- Matrix / MFMA accumulators and operations -------------------------

    def mfma_acc(
        self,
        form: str,
        init=0,
        *,
        acc: str | None = None,
        agpr: bool | None = None,
        name: str | None = None,
        at: int | None = None,
        same_as=None,
    ) -> TypedValue:
        """Allocates a matrix accumulator for ``form`` in ``'a'`` (AGPR) or ``'v'`` (VGPR)."""
        self._at()
        y = None
        if same_as is not None:
            y = getattr(same_as, "_typed", None)
            if y is None:
                raise ValueError(f"same_as={same_as!r}: a whole TypedValue")
        x = None if init is None else leaf(init)
        out = typed(_native.builder_mfma_acc(self._n, form, acc, agpr, name, at, y, x))
        if at is not None or same_as is not None:
            self._sync_values(links=True)
        return out

    def mfma(
        self,
        form: str,
        a,
        b,
        c=0,
        *,
        dst=None,
        acc: str | None = None,
        agpr: bool | None = None,
        name: str | None = None,
        cbsz: int = 0,
        abid: int = 0,
        blgp: int = 0,
        neg=None,
        comment: str = "",
        tag=(),
        abl=(),
    ) -> TypedValue:
        """Emits a dense MFMA instruction and returns the destination accumulator."""
        self._at()
        c_acc = _maybe_acc_in(c)
        c_arg = None if c is None else operand(c)
        dst_in = None if dst is None else _acc_in(dst)
        cfg = (c_arg, c_acc, dst_in, acc, agpr, name, list(abl))
        opts = _mfma_opts(cbsz, abid, blgp, neg, comment, tag, ())
        return typed(_native.builder_mfma(self._n, form, operand(a), operand(b), cfg, opts))

    def smfmac(
        self,
        form: str,
        a,
        b,
        idx,
        c,
        *,
        acc: str | None = None,
        agpr: bool | None = None,
        cbsz: int = 0,
        abid: int = 0,
        comment: str = "",
        tag=(),
        abl=(),
    ) -> TypedValue:
        """Emits a sparse 4:2 SMFMAC instruction updating ``c`` in-place."""
        self._at()
        opts = _mfma_opts(cbsz, abid, 0, None, comment, tag, ())
        return typed(
            _native.builder_smfmac(
                self._n,
                form,
                operand(a),
                operand(b),
                operand(idx),
                _acc_in(c),
                acc,
                agpr,
                list(abl),
                opts,
            )
        )

    def read_acc(
        self,
        src,
        *,
        dst=None,
        copy: bool = False,
        name: str | None = None,
    ) -> TypedValue:
        """Returns ``src`` in VGPRs, emitting ``v_accvgpr_read_b32`` when ``src`` is in AGPRs."""
        self._at()
        dst_in = None if dst is None else _acc_in(dst)
        return typed(_native.builder_read_acc(self._n, _acc_in(src), dst_in, copy, name))

    def write_acc(
        self,
        src,
        *,
        dst=None,
        acc: str | None = None,
        agpr: bool | None = None,
        copy: bool = False,
        name: str | None = None,
    ) -> TypedValue:
        """Writes ``src`` into an ``'a'`` (AGPR) or ``'v'`` (VGPR) accumulator."""
        self._at()
        dst_in = None if dst is None else _acc_in(dst)
        return typed(
            _native.builder_write_acc(self._n, _acc_in(src), dst_in, acc, agpr, copy, name)
        )


def _acc_in(x) -> tuple:
    if isinstance(x, TypedValue):
        return ("t", x._typed)
    r = as_reg(x)
    if isinstance(r, VReg):
        return ("r", native_reg(r))
    raise TypeError(f"{x!r}: expected an accumulator TypedValue or VReg")


def _maybe_acc_in(x) -> tuple | None:
    if isinstance(x, TypedValue):
        return ("t", x._typed)
    r = as_reg(x)
    if isinstance(r, VReg):
        return ("r", native_reg(r))
    return None


def _claim(x) -> tuple:
    r = PHYS[x] if isinstance(x, str) else as_reg(x)
    if isinstance(r, Phys):
        return ("p", r.name)
    return ("r", native_reg(r))


def _through(x) -> str:
    if isinstance(x, str):
        return x
    r = as_reg(x)
    return r.value.name


def _join(into: dict, other: dict) -> None:
    for k, v in other.items():
        join = getattr(v, "join", None)
        if join is None:
            continue
        into[k] = v.join(into[k]) if into.get(k) is not None else copy.copy(v)


def _deferred(b, d) -> tuple:
    kind, stem, label, top, starts, last, region = d
    if kind == "exit":
        return ("exit", {"exit": b._at_row(starts[0]), "upto": b._at_row(last), "region": region})
    rows = [None if s is None else b._at_row(s) for s in starts]
    return (
        "peel",
        {"stem": stem, "label": label, "max": top, "starts": rows, "last": b._at_row(last)},
    )
