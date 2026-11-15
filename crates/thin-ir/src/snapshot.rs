//! The program snapshot: an immutable view of a stream that the analyses
//! read, with its cache of shared results.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, OnceLock};

use thin_target::Target;

use crate::cfg::{cfg_of, Cfg, Taken};
use crate::{exec, Error, FieldId, FieldValue, Instr, Operand, Reg, Request};

/// A wave class: the resolved skip branches, or none for every wave.
pub type WaveClass = Option<Vec<(usize, bool)>>;

/// A group of waves that take the same skip branches: its waves, and per
/// tagged branch whether they take it.
pub type WaveGroup = (Vec<u32>, Vec<(usize, bool)>);

/// What the snapshot holds of a value.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ValueInfo {
    /// Its register file letter.
    pub file: char,
    /// Its dwords.
    pub n: u32,
    /// Its name, interned.
    pub name: u32,
    /// Its first physical register, once known.
    pub phys: Option<u32>,
    /// The register the writer or the ABI fixed.
    pub pin: Option<u32>,
    /// The value whose registers it shares.
    pub same_as: Option<u32>,
    /// Its alignment in dwords.
    pub align: u32,
    /// Its position in its register-file group.
    pub index: u32,
    /// The order in which its name first appears among its file's values.
    pub group: u32,
    /// Whether its partial-EXEC writes cover the lanes its reads use.
    pub covered: bool,
}

/// What the snapshot holds of an LDS region.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    /// Its first byte.
    pub base: u64,
    /// Its bytes.
    pub size: u64,
    /// Its name.
    pub name: Arc<str>,
    /// Its parent's name, or its own when it has no parent.
    pub leaf: Arc<str>,
    /// Whether it is a region of this kernel's layout.
    pub on_layout: bool,
    /// Its tags.
    pub tags: Vec<Arc<str>>,
}

/// A value as a kernel gives it to the snapshot.
#[derive(Clone, Debug, Default)]
pub struct GatherValue {
    /// Its register file letter.
    pub file: char,
    /// Its dwords.
    pub n: u32,
    /// Its alignment.
    pub align: u32,
    /// Its name, which is its group's.
    pub name: String,
    /// Its pin.
    pub pin: Option<u32>,
    /// The value whose registers it shares.
    pub same_as: Option<u32>,
    /// Its position in its group.
    pub index: u32,
    /// Its hand-laid parts' names, in part-number order (1, 2, …).
    pub parts: Vec<String>,
    /// Its own first register in the file's assignment, if any.
    pub register: Option<u32>,
}

/// A register file as a kernel gives it to the snapshot.
#[derive(Clone, Debug, Default)]
pub struct GatherFile {
    /// Its letter.
    pub letter: char,
    /// The lanes-covered names.
    pub lanes: Vec<String>,
    /// The groups it entered, in order.
    pub entered: Vec<String>,
    /// Its pinned groups, in order.
    pub pinned: Vec<String>,
    /// Whether its assignment is placed.
    pub placed: bool,
}

/// An LDS region as a kernel gives it to the snapshot.
#[derive(Clone, Debug, Default)]
pub struct GatherRegion {
    /// Its identity, as the accesses name it.
    pub key: u64,
    /// Its first byte.
    pub base: u64,
    /// Its bytes.
    pub size: u64,
    /// Its name.
    pub name: String,
    /// Its parent's name, if it has a parent.
    pub parent: Option<String>,
    /// Whether it is a layout region of this kernel's layout.
    pub on_layout: bool,
    /// Its `tags()`, or none.
    pub tags: Vec<String>,
}

/// What a snapshot gathers from a kernel, besides its instructions.
#[derive(Clone, Debug, Default)]
pub struct Gather {
    /// The values, numbered by position.
    pub values: Vec<GatherValue>,
    /// The register files, in order v, s, a (those the kernel has).
    pub files: Vec<GatherFile>,
    /// The LDS regions the instructions' accesses name.
    pub regions: Vec<GatherRegion>,
    /// The SGPR holding the wave's index in the workgroup.
    pub wave_reg: Option<Reg>,
    /// Waves per workgroup.
    pub nwaves: u32,
    /// Whether the passes treat each wave class apart.
    pub by_wave: bool,
}

/// The cached shared analyses.
#[derive(Debug, Default)]
struct Cache {
    cfgs: Mutex<HashMap<WaveClass, Arc<Cfg>>>,
    exec: OnceLock<Arc<Vec<Option<u64>>>>,
}

/// An immutable native view of a stream.
#[derive(Debug)]
pub struct Snapshot {
    target: &'static Target,
    names: Vec<Arc<str>>,
    syms: HashMap<Arc<str>, u32>,
    values: Vec<ValueInfo>,
    instrs: Arc<[Arc<Instr>]>,
    entry: Vec<Reg>,
    through: HashMap<(i32, u32), (u32, u32)>,
    wave_reg: Option<Reg>,
    nwaves: u32,
    by_wave: bool,
    regions: HashMap<u64, Region>,
    requests: BTreeMap<usize, (usize, Request)>,
    control: Option<crate::control::ControlFlow>,
    cache: Cache,
}

impl Snapshot {
    /// A bare snapshot: no entry, wave register, regions or requests, no
    /// waves and no wave classes.
    pub fn new(
        target: &'static Target,
        values: Vec<ValueInfo>,
        instrs: Arc<[Arc<Instr>]>,
    ) -> Snapshot {
        let mut s = Snapshot {
            target,
            names: Vec::new(),
            syms: HashMap::new(),
            values,
            instrs,
            entry: Vec::new(),
            through: HashMap::new(),
            wave_reg: None,
            nwaves: 0,
            by_wave: false,
            regions: HashMap::new(),
            requests: BTreeMap::new(),
            control: None,
            cache: Cache::default(),
        };
        s.intern("");
        s
    }

    /// The snapshot of a free list of instructions whose registers are
    /// physical: value k is the value numbered k, in the file its registers
    /// name, at register 0, named "".
    pub fn of_list(target: &'static Target, instrs: Arc<[Arc<Instr>]>) -> Result<Snapshot, Error> {
        let mut values: Vec<ValueInfo> = Vec::new();
        for r in instrs.iter().flat_map(|i| registers(i)) {
            let k = usize::try_from(r.value)
                .map_err(|_| Error::Value("a value numbered below 0".into()))?;
            if values.len() <= k {
                values.resize(k + 1, ValueInfo::default());
            }
            if values[k].phys.is_none() {
                values[k] = ValueInfo {
                    file: r.file,
                    phys: Some(0),
                    align: 1,
                    ..ValueInfo::default()
                };
            }
        }
        let mut s = Snapshot::new(target, values, instrs);
        s.requests = requests(target, &s.instrs)?;
        Ok(s)
    }

    /// The snapshot of a kernel's instructions and what `g` gathers of it
    ///
    pub fn of_kernel(
        target: &'static Target,
        g: &Gather,
        instrs: Arc<[Arc<Instr>]>,
    ) -> Result<Snapshot, Error> {
        let mut s = Snapshot::new(target, Vec::new(), instrs);
        s.values = gather_values(&mut s, g);
        s.through = gather_through(&mut s, g);
        s.entry = gather_entry(g);
        s.wave_reg = g.wave_reg;
        s.nwaves = g.nwaves;
        s.by_wave = g.by_wave;
        s.validate(
            &s.entry
                .iter()
                .chain(&g.wave_reg)
                .copied()
                .collect::<Vec<_>>(),
        )?;
        for r in &g.regions {
            let region = Region {
                base: r.base,
                size: r.size,
                name: r.name.as_str().into(),
                leaf: r.parent.as_deref().unwrap_or(&r.name).into(),
                on_layout: r.on_layout,
                tags: r.tags.iter().map(|t| t.as_str().into()).collect(),
            };
            s.regions.insert(r.key, region);
        }
        s.requests = requests(target, &s.instrs)?;
        Ok(s)
    }

    /// Sets explicit entry subranges for a physical program. Pinning a
    /// physical register is not itself a claim that hardware initializes it.
    pub fn with_entry(mut self, entry: Vec<Reg>) -> Result<Snapshot, Error> {
        self.validate(&entry)?;
        for r in &entry {
            let value = &self.values[r.value as usize];
            if r.file != value.file || r.lo.checked_add(r.n).is_none_or(|n| n > value.n) {
                return Err(Error::Value(
                    "entry register outside its physical value".into(),
                ));
            }
        }
        self.entry = entry;
        Ok(self)
    }

    /// Installs typed physical transfers and invalidates cached analyses.
    /// Unresolved transfers have only their explicitly known edges, not an
    /// invented fall-through. The caller must inspect completeness before
    /// interpreting any result as a whole-program proof.
    pub fn with_control_flow(
        mut self,
        control: Option<crate::control::ControlFlow>,
    ) -> Result<Self, Error> {
        if let Some(flow) = &control {
            // The instructions' registers once; each link alone below.
            self.validate(&[])?;
            let n = self.instrs.len();
            let mut previous = None;
            for transfer in &flow.transfers {
                if transfer.instruction >= n || previous.is_some_and(|p| p >= transfer.instruction)
                {
                    return Err(Error::Value(
                        "invalid physical transfer order or row".into(),
                    ));
                }
                let link = match transfer.kind {
                    crate::control::TransferKind::Call { link, continuation } => {
                        if matches!(continuation, crate::control::Destination::Instruction(row) if row >= n)
                        {
                            return Err(Error::Value(
                                "physical continuation outside the program".into(),
                            ));
                        }
                        Some(link)
                    }
                    crate::control::TransferKind::Return { link } => Some(link),
                    crate::control::TransferKind::UnresolvedCall { continuation } => {
                        if matches!(continuation, Some(crate::control::Destination::Instruction(row)) if row >= n)
                        {
                            return Err(Error::Value(
                                "physical continuation outside the program".into(),
                            ));
                        }
                        None
                    }
                    crate::control::TransferKind::Jump => None,
                };
                if let Some(link) = link {
                    if !self.known(&link) {
                        return Err(Error::Value(
                            "a register of a value not in the program".into(),
                        ));
                    }
                    let value = &self.values[link.value as usize];
                    if link.file != value.file
                        || link.lo.checked_add(link.n).is_none_or(|end| end > value.n)
                    {
                        return Err(Error::Value("physical link outside its value".into()));
                    }
                }
                for edge in &transfer.edges {
                    if matches!(edge.target, crate::control::Destination::Instruction(row) if row >= n)
                    {
                        return Err(Error::Value("physical edge outside the program".into()));
                    }
                    if matches!(edge.kind, crate::control::EdgeKind::Return { caller } if caller >= n)
                    {
                        return Err(Error::Value(
                            "physical return names an invalid caller".into(),
                        ));
                    }
                }
                previous = Some(transfer.instruction);
            }
            for transfer in &flow.transfers {
                for edge in &transfer.edges {
                    if let crate::control::EdgeKind::Return { caller } = edge.kind {
                        let call = flow
                            .transfers
                            .binary_search_by_key(&caller, |t| t.instruction)
                            .ok()
                            .map(|k| &flow.transfers[k]);
                        if !matches!(call.map(|c| c.kind), Some(crate::control::TransferKind::Call { continuation, .. }) if continuation == edge.target)
                        {
                            return Err(Error::Value(
                                "return edge is not its caller's continuation".into(),
                            ));
                        }
                    }
                }
            }
        }
        self.control = control;
        self.cache = Cache::default();
        Ok(self)
    }

    /// Explicit physical transfers, if present.
    pub fn control_flow(&self) -> Option<&crate::control::ControlFlow> {
        self.control.as_ref()
    }

    /// The same snapshot for a workgroup of `nwaves` waves, its wave
    /// classes treated apart when `by_wave` is set.
    pub fn with_waves(mut self, nwaves: u32, by_wave: bool) -> Snapshot {
        self.nwaves = nwaves;
        self.by_wave = by_wave;
        self.cache = Cache::default();
        self
    }

    /// Whether a register names one of the values.
    fn known(&self, r: &Reg) -> bool {
        usize::try_from(r.value).is_ok_and(|v| v < self.values.len())
    }

    /// A register naming a value past the values, or below 0, is a
    /// `ValueError`: the instructions' registers and `more`.
    fn validate(&self, more: &[Reg]) -> Result<(), Error> {
        let nvalues = self.values.len() as i64;
        let mut regs = self
            .instrs
            .iter()
            .flat_map(|i| registers(i))
            .chain(more.iter().copied());
        if regs.any(|r| r.value < 0 || r.value as i64 >= nvalues) {
            return Err(Error::Value(
                "a register of a value not in the program".into(),
            ));
        }
        Ok(())
    }

    /// The number of a string, interning it.
    pub fn intern(&mut self, name: &str) -> u32 {
        if let Some(&k) = self.syms.get(name) {
            return k;
        }
        let k = self.names.len() as u32;
        let name: Arc<str> = name.into();
        self.names.push(name.clone());
        self.syms.insert(name, k);
        k
    }

    /// The string of an interned number.
    pub fn name(&self, sym: u32) -> &str {
        &self.names[sym as usize]
    }

    /// The target.
    pub fn target(&self) -> &'static Target {
        self.target
    }

    /// The values, by number.
    pub fn values(&self) -> &[ValueInfo] {
        &self.values
    }

    /// The instructions, in layout order.
    pub fn instrs(&self) -> &Arc<[Arc<Instr>]> {
        &self.instrs
    }

    /// The registers the hardware fills at entry.
    pub fn entry(&self) -> &[Reg] {
        &self.entry
    }

    /// The SGPR holding the wave's index in the workgroup.
    pub fn wave_reg(&self) -> Option<Reg> {
        self.wave_reg
    }

    /// Waves per workgroup.
    pub fn nwaves(&self) -> u32 {
        self.nwaves
    }

    /// Whether the passes treat each wave class apart.
    pub fn by_wave(&self) -> bool {
        self.by_wave
    }

    /// Per wait marker's index, its counter and request.
    pub fn requests(&self) -> &BTreeMap<usize, (usize, Request)> {
        &self.requests
    }

    /// A register's physical number; its value must be placed.
    pub fn phys(&self, r: &Reg) -> u32 {
        self.values[r.value as usize].phys.expect("a placed value") + r.lo
    }

    /// A register's file letter.
    pub fn file(&self, r: &Reg) -> char {
        self.values[r.value as usize].file
    }

    /// The name (as interned) a register goes through.
    pub fn through(&self, r: &Reg) -> u32 {
        self.through.get(&(r.value, r.part)).map_or(0, |t| t.0)
    }

    /// The alias (as interned) of the name a register goes through.
    pub fn alias(&self, r: &Reg) -> u32 {
        self.through.get(&(r.value, r.part)).map_or(0, |t| t.1)
    }

    /// `<f><p>` for one dword, else `<f>[<p>:<p+n−1>]`.
    pub fn reg_text(&self, r: &Reg) -> String {
        crate::print::phys_text(self.file(r), self.phys(r), r.n)
    }

    /// The region of an LDS access, by its identity.
    pub fn region(&self, key: u64) -> Option<&Region> {
        self.regions.get(&key)
    }

    /// The region instruction k accesses.
    pub fn region_of(&self, k: usize) -> Option<&Region> {
        let access = self.instrs[k].attrs().lds_access.as_ref()?;
        self.region(access.region)
    }

    /// Instruction k's Places names, then its region's tags, each once.
    pub fn answers(&self, k: usize) -> Vec<Arc<str>> {
        let mut out = self.instrs[k].attrs().places.clone();
        out.extend(
            self.region_of(k)
                .into_iter()
                .flat_map(|r| r.tags.iter().cloned()),
        );
        // A name both placed and a region tag answers once.
        let mut seen = std::collections::HashSet::new();
        out.retain(|a| seen.insert(a.clone()));
        out
    }

    /// The same snapshot with instruction k replaced by an `ir.comment`
    /// saying it was muted, and a fresh cache. The comment is a valid
    /// instruction, so the muted snapshot still verifies.
    pub fn muted(&self, k: usize) -> Snapshot {
        let comment = self.target.pseudo(thin_target::Pseudo::Comment);
        let mut instrs = self.instrs.to_vec();
        let text = (FieldId::of("text"), FieldValue::Text("; muted".into()));
        instrs[k] = Arc::new(Instr::new(comment, Vec::new(), vec![text], None));
        let mut requests = self.requests.clone();
        requests.remove(&k);
        Snapshot {
            target: self.target,
            names: self.names.clone(),
            syms: self.syms.clone(),
            values: self.values.clone(),
            instrs: instrs.into(),
            entry: self.entry.clone(),
            through: self.through.clone(),
            wave_reg: self.wave_reg,
            nwaves: self.nwaves,
            by_wave: self.by_wave,
            regions: self.regions.clone(),
            requests,
            control: self.control.clone(),
            cache: Cache::default(),
        }
    }

    /// The waves grouped by the skip branches they take: each group's
    /// waves, and per tagged branch the first wave's bit.
    pub fn wave_groups(&self) -> Vec<WaveGroup> {
        let tagged: Vec<(usize, u64)> = (self.instrs.iter().enumerate())
            .filter_map(|(k, i)| i.attrs().waves_taken.map(|m| (k, m)))
            .collect();
        if tagged.is_empty() {
            return Vec::new();
        }
        let mut groups: Vec<(Vec<bool>, Vec<u32>)> = Vec::new();
        for w in 0..self.nwaves {
            let sig: Vec<bool> = tagged
                .iter()
                .map(|&(_, m)| w < 64 && m >> w & 1 == 1)
                .collect();
            match groups.iter_mut().find(|(s, _)| *s == sig) {
                Some((_, waves)) => waves.push(w),
                None => groups.push((sig, vec![w])),
            }
        }
        (groups.into_iter())
            .map(|(sig, waves)| (waves, tagged.iter().map(|&(k, _)| k).zip(sig).collect()))
            .collect()
    }

    /// The wave classes the passes treat apart: the groups' maps when
    /// `by_wave` is set and there are groups, else one class for every
    /// wave.
    pub fn wave_classes(&self) -> Vec<WaveClass> {
        let groups = self.wave_groups();
        if self.by_wave && !groups.is_empty() {
            groups.into_iter().map(|(_, taken)| Some(taken)).collect()
        } else {
            vec![None]
        }
    }

    /// The CFG without label blocks for a wave class, built once per class.
    pub fn cfg(&self, class: Option<&Taken>) -> Arc<Cfg> {
        let key: WaveClass = class.map(|c| {
            let mut c = c.to_vec();
            c.sort_unstable();
            c
        });
        let mut cfgs = self.cache.cfgs.lock().expect("an unpoisoned cache");
        let cfg = cfgs.entry(key).or_insert_with(|| {
            let ordinary = cfg_of(&self.instrs, class, false);
            let Some(control) = &self.control else {
                return Arc::new(ordinary);
            };
            let mut successors = ordinary.succ;
            for transfer in &control.transfers {
                successors[transfer.instruction] = transfer
                    .edges
                    .iter()
                    .filter_map(|edge| match edge.target {
                        crate::control::Destination::Instruction(row) => Some(row),
                        crate::control::Destination::Exit => None,
                    })
                    .collect();
                successors[transfer.instruction].sort_unstable();
                successors[transfer.instruction].dedup();
            }
            Arc::new(Cfg::from_succ_only(successors))
        });
        cfg.clone()
    }

    /// EXEC on entry to each instruction, computed once.
    pub fn exec(&self) -> Arc<Vec<Option<u64>>> {
        self.cache
            .exec
            .get_or_init(|| Arc::new(exec::exec(self, &self.cfg(None))))
            .clone()
    }
}

/// An instruction's register operands and WaitUses registers.
fn registers(i: &Instr) -> impl Iterator<Item = Reg> + '_ {
    let wait_uses = i
        .attrs()
        .wait_uses
        .iter()
        .flat_map(|(_, regs)| regs.iter().copied());
    i.ops()
        .iter()
        .filter_map(Operand::reg)
        .copied()
        .chain(wait_uses)
}

/// The request of every wait marker, by index.
fn requests(
    target: &Target,
    instrs: &[Arc<Instr>],
) -> Result<BTreeMap<usize, (usize, Request)>, Error> {
    let mut out = BTreeMap::new();
    for (k, i) in instrs.iter().enumerate() {
        if let Some(r) = i.request(target)? {
            out.insert(k, r);
        }
    }
    Ok(out)
}

/// The file a gathered value is in.
fn file_of(g: &Gather, letter: char) -> Option<&GatherFile> {
    g.files.iter().find(|f| f.letter == letter)
}

/// The values' snapshot records: names interned, groups numbered by first
/// appearance in their file, `covered` from the file's lanes, `phys` from
/// the placement or the `same_as` root.
fn gather_values(s: &mut Snapshot, g: &Gather) -> Vec<ValueInfo> {
    let mut groups: HashMap<(char, &str), u32> = HashMap::new();
    let mut counts: HashMap<char, u32> = HashMap::new();
    let mut out = Vec::with_capacity(g.values.len());
    for (k, v) in g.values.iter().enumerate() {
        let count = counts.entry(v.file).or_default();
        let group = *groups.entry((v.file, &v.name)).or_insert_with(|| {
            *count += 1;
            *count - 1
        });
        let file = file_of(g, v.file);
        let stem = v.name.split('#').next().unwrap_or("");
        let covered = file.is_some_and(|f| f.lanes.iter().any(|l| *l == v.name || l == stem));
        let placed = file.is_some_and(|f| f.placed);
        let phys = if placed {
            v.register
        } else {
            g.values[root(g, k)].register
        };
        out.push(ValueInfo {
            file: v.file,
            n: v.n,
            name: s.intern(&v.name),
            phys,
            pin: v.pin,
            same_as: v.same_as,
            align: v.align,
            index: v.index,
            group,
            covered,
        });
    }
    out
}

/// The number of the end of value k's `same_as` chain.
fn root(g: &Gather, k: usize) -> usize {
    let mut at = k;
    for _ in 0..g.values.len() {
        match g.values[at].same_as {
            Some(next) => at = next as usize,
            None => break,
        }
    }
    at
}

/// The (name, alias) of each value's own name and hand-laid parts.
fn gather_through(s: &mut Snapshot, g: &Gather) -> HashMap<(i32, u32), (u32, u32)> {
    let alias = |file: char, t: &str| -> String {
        let first = g.values.iter().position(|v| v.file == file && v.name == t);
        first.map_or(t.to_owned(), |k| g.values[root(g, k)].name.clone())
    };
    let mut out = HashMap::new();
    for (k, v) in g.values.iter().enumerate() {
        let names = std::iter::once(v.name.clone())
            .chain(v.parts.iter().map(|p| format!("{}.{p}", v.name)));
        for (j, name) in names.enumerate() {
            let pair = (s.intern(&name), s.intern(&alias(v.file, &name)));
            out.insert((k as i32, j as u32), pair);
        }
    }
    out
}

/// The entry registers: per file in order, its entered groups and then
/// its pinned groups, each once, each value of a group whole.
fn gather_entry(g: &Gather) -> Vec<Reg> {
    let mut out = Vec::new();
    for letter in ['v', 's', 'a'] {
        let Some(f) = file_of(g, letter) else {
            continue;
        };
        let mut seen: Vec<&str> = Vec::new();
        for name in f.entered.iter().chain(&f.pinned) {
            if seen.contains(&name.as_str()) {
                continue;
            }
            seen.push(name);
            let group =
                (g.values.iter().enumerate()).filter(|(_, v)| v.file == letter && v.name == *name);
            out.extend(group.map(|(k, v)| Reg::new(k as i32, letter, 0, v.n)));
        }
    }
    out
}
