//! Attributes: what layers and passes add to an instruction.

use std::sync::Arc;

use thin_target::Flags;

use crate::field::{HostObject, Tag};
use crate::operand::Reg;
use crate::Error;

/// An attribute class, in the verifier's order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AttrClass {
    /// A note printed after the line.
    Comment,
    /// Dropped when the build ablates one of these classes.
    Ablation,
    /// Reads the alias check must not report.
    AliasOk,
    /// The writer's marker for wait bookkeeping.
    Tag,
    /// An operand staged into a scratch SGPR before the instruction.
    Staged,
    /// Registers a wait just before this instruction named.
    WaitUses,
    /// The lane region the instruction is in.
    InRegion,
    /// Placed by the hazard pass.
    Hazard,
    /// Names an op that writes no register answers to.
    Places,
    /// A device-scope atomic the coherence check must not judge.
    Published,
    /// An LDS access the layout placed.
    LdsAccess,
    /// An M0 write of a constant a DMA made ahead of itself.
    M0Set,
    /// The waves that take this skip branch.
    WavesTaken,
    /// An LDS fence.
    Fence,
}

impl AttrClass {
    /// Every class, in the verifier's order.
    pub const ALL: [AttrClass; 14] = [
        AttrClass::Comment,
        AttrClass::Ablation,
        AttrClass::AliasOk,
        AttrClass::Tag,
        AttrClass::Staged,
        AttrClass::WaitUses,
        AttrClass::InRegion,
        AttrClass::Hazard,
        AttrClass::Places,
        AttrClass::Published,
        AttrClass::LdsAccess,
        AttrClass::M0Set,
        AttrClass::WavesTaken,
        AttrClass::Fence,
    ];

    /// The class's name.
    pub fn name(self) -> &'static str {
        const NAMES: [&str; 14] = [
            "Comment",
            "Ablation",
            "AliasOk",
            "Tag",
            "Staged",
            "WaitUses",
            "InRegion",
            "Hazard",
            "Places",
            "Published",
            "LdsAccess",
            "M0Set",
            "WavesTaken",
            "Fence",
        ];
        NAMES[self as usize]
    }

    /// The class of that name.
    pub fn named(name: &str) -> Option<AttrClass> {
        AttrClass::ALL.into_iter().find(|c| c.name() == name)
    }

    /// The flags of the forms it may sit on; `NONE` is any form.
    pub fn on(self) -> Flags {
        match self {
            AttrClass::Places | AttrClass::LdsAccess => Flags::VMEM | Flags::DS,
            AttrClass::Published => Flags::ATOMIC,
            AttrClass::WavesTaken => Flags::BRANCH,
            AttrClass::Fence => Flags::BARRIER,
            _ => Flags::NONE,
        }
    }
}

/// An LDS access the layout placed: the object put, and what the passes
/// read of it.
#[derive(Clone, Debug)]
pub struct LdsAccess {
    /// The attribute object itself.
    pub object: HostObject,
    /// The region's identity.
    pub region: u64,
    /// The access's field.
    pub field: i128,
    /// What it accesses.
    pub what: Arc<str>,
    /// Its ordering.
    pub ordered: Arc<str>,
}

/// One attribute's data, as it is put.
#[derive(Clone, Debug)]
pub enum Attr {
    /// `Comment(text)`.
    Comment(Arc<str>),
    /// `Ablation(classes)`, in any order.
    Ablation(Vec<Arc<str>>),
    /// `AliasOk(pairs)` of (read, written or none), in any order.
    AliasOk(Vec<(Arc<str>, Option<Arc<str>>)>),
    /// `Tag(value)`.
    Tag(Tag),
    /// `Staged(stage)`: the object put.
    Staged(HostObject),
    /// `WaitUses(regs)`: the object put and its registers.
    WaitUses(HostObject, Vec<Reg>),
    /// `InRegion(id)`.
    InRegion(i128),
    /// `Hazard()`.
    Hazard,
    /// `Places(names)`.
    Places(Vec<Arc<str>>),
    /// `Published(why)`.
    Published(Arc<str>),
    /// `LdsAccess(…)`.
    LdsAccess(LdsAccess),
    /// `M0Set(value)`.
    M0Set(i128),
    /// `WavesTaken(waves)`, each below 64.
    WavesTaken(Vec<i128>),
    /// `Fence()`.
    Fence,
}

impl Attr {
    /// Its class.
    pub fn class(&self) -> AttrClass {
        match self {
            Attr::Comment(_) => AttrClass::Comment,
            Attr::Ablation(_) => AttrClass::Ablation,
            Attr::AliasOk(_) => AttrClass::AliasOk,
            Attr::Tag(_) => AttrClass::Tag,
            Attr::Staged(_) => AttrClass::Staged,
            Attr::WaitUses(..) => AttrClass::WaitUses,
            Attr::InRegion(_) => AttrClass::InRegion,
            Attr::Hazard => AttrClass::Hazard,
            Attr::Places(_) => AttrClass::Places,
            Attr::Published(_) => AttrClass::Published,
            Attr::LdsAccess(_) => AttrClass::LdsAccess,
            Attr::M0Set(_) => AttrClass::M0Set,
            Attr::WavesTaken(_) => AttrClass::WavesTaken,
            Attr::Fence => AttrClass::Fence,
        }
    }
}

/// An instruction's attributes, each stored as table says.
#[derive(Clone, Debug, Default)]
pub struct Attrs {
    /// The comment; never empty.
    pub comment: Option<Arc<str>>,
    /// The ablation classes, sorted.
    pub ablation: Vec<Arc<str>>,
    /// The alias pairs, sorted.
    pub alias_ok: Vec<(Arc<str>, Option<Arc<str>>)>,
    /// The tag; never empty.
    pub tag: Option<Tag>,
    /// Whether the tag starts with `"noorder"`.
    pub noorder: bool,
    /// The staged object.
    pub staged: Option<HostObject>,
    /// The wait-uses object and its registers.
    pub wait_uses: Option<(HostObject, Vec<Reg>)>,
    /// The lane region.
    pub in_region: Option<i128>,
    /// Whether the hazard pass placed it.
    pub hazard: bool,
    /// The names it answers to.
    pub places: Vec<Arc<str>>,
    /// Why it is published.
    pub published: Option<Arc<str>>,
    /// The LDS access.
    pub lds_access: Option<LdsAccess>,
    /// The constant of the M0 write.
    pub m0_set: Option<i128>,
    /// The mask of waves that take the branch.
    pub waves_taken: Option<u64>,
    /// Whether it is an LDS fence.
    pub fence: bool,
}

impl Attrs {
    /// Replaces the attribute of `attr`'s class. Empty data removes it,
    /// as the table's "absent when" column says; a wave of 64 or more is a
    /// `ValueError`.
    pub fn put(&mut self, attr: Attr) -> Result<(), Error> {
        if let Attr::WavesTaken(waves) = &attr {
            wave_mask(waves)?;
        }
        self.drop(attr.class());
        match attr {
            Attr::Comment(t) => self.comment = (!t.is_empty()).then_some(t),
            Attr::Ablation(mut v) => self.ablation = sorted(&mut v),
            Attr::AliasOk(mut v) => self.alias_ok = sorted(&mut v),
            Attr::Tag(t) => {
                self.noorder = matches!(t.first(), Some(crate::Atom::Str(s)) if &**s == "noorder");
                self.tag = (!t.is_empty()).then_some(t);
            }
            Attr::Staged(o) => self.staged = Some(o),
            Attr::WaitUses(o, regs) => self.wait_uses = Some((o, regs)),
            Attr::InRegion(id) => self.in_region = Some(id),
            Attr::Hazard => self.hazard = true,
            Attr::Places(v) => self.places = v,
            Attr::Published(why) => self.published = Some(why),
            Attr::LdsAccess(a) => self.lds_access = Some(a),
            Attr::M0Set(v) => self.m0_set = Some(v),
            Attr::WavesTaken(waves) => self.waves_taken = Some(wave_mask(&waves)?),
            Attr::Fence => self.fence = true,
        }
        Ok(())
    }

    /// Removes the attribute of that class.
    pub fn drop(&mut self, class: AttrClass) {
        match class {
            AttrClass::Comment => self.comment = None,
            AttrClass::Ablation => self.ablation.clear(),
            AttrClass::AliasOk => self.alias_ok.clear(),
            AttrClass::Tag => (self.tag, self.noorder) = (None, false),
            AttrClass::Staged => self.staged = None,
            AttrClass::WaitUses => self.wait_uses = None,
            AttrClass::InRegion => self.in_region = None,
            AttrClass::Hazard => self.hazard = false,
            AttrClass::Places => self.places.clear(),
            AttrClass::Published => self.published = None,
            AttrClass::LdsAccess => self.lds_access = None,
            AttrClass::M0Set => self.m0_set = None,
            AttrClass::WavesTaken => self.waves_taken = None,
            AttrClass::Fence => self.fence = false,
        }
    }

    /// The attribute of that class, if present.
    pub fn get(&self, class: AttrClass) -> Option<Attr> {
        let present = |b: bool, a: Attr| b.then_some(a);
        match class {
            AttrClass::Comment => self.comment.clone().map(Attr::Comment),
            AttrClass::Ablation => present(
                !self.ablation.is_empty(),
                Attr::Ablation(self.ablation.clone()),
            ),
            AttrClass::AliasOk => present(
                !self.alias_ok.is_empty(),
                Attr::AliasOk(self.alias_ok.clone()),
            ),
            AttrClass::Tag => self.tag.clone().map(Attr::Tag),
            AttrClass::Staged => self.staged.clone().map(Attr::Staged),
            AttrClass::WaitUses => self.wait_uses.clone().map(|(o, r)| Attr::WaitUses(o, r)),
            AttrClass::InRegion => self.in_region.map(Attr::InRegion),
            AttrClass::Hazard => present(self.hazard, Attr::Hazard),
            AttrClass::Places => {
                present(!self.places.is_empty(), Attr::Places(self.places.clone()))
            }
            AttrClass::Published => self.published.clone().map(Attr::Published),
            AttrClass::LdsAccess => self.lds_access.clone().map(Attr::LdsAccess),
            AttrClass::M0Set => self.m0_set.map(Attr::M0Set),
            AttrClass::WavesTaken => (self.waves_taken).map(|m| {
                Attr::WavesTaken(
                    (0..64)
                        .filter(|w| m >> w & 1 == 1)
                        .map(i128::from)
                        .collect(),
                )
            }),
            AttrClass::Fence => present(self.fence, Attr::Fence),
        }
    }

    /// Whether the attribute of that class is present.
    pub fn has(&self, class: AttrClass) -> bool {
        match class {
            AttrClass::Ablation => !self.ablation.is_empty(),
            AttrClass::AliasOk => !self.alias_ok.is_empty(),
            AttrClass::Places => !self.places.is_empty(),
            AttrClass::Hazard => self.hazard,
            AttrClass::Fence => self.fence,
            AttrClass::WaitUses => self.wait_uses.is_some(),
            AttrClass::LdsAccess => self.lds_access.is_some(),
            AttrClass::Staged => self.staged.is_some(),
            other => self.get(other).is_some(),
        }
    }
}

/// The items sorted, moved out.
fn sorted<T: Ord>(v: &mut Vec<T>) -> Vec<T> {
    v.sort();
    std::mem::take(v)
}

/// The 64-bit mask of a set of waves; a wave past 63 is a `ValueError`.
fn wave_mask(waves: &[i128]) -> Result<u64, Error> {
    waves.iter().try_fold(0, |m, &w| {
        if (0..64).contains(&w) {
            Ok(m | 1 << w)
        } else {
            Err(Error::Value(format!(
                "WavesTaken: wave {w} (a workgroup has 64 at most)"
            )))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_data_is_absent() {
        let mut a = Attrs::default();
        a.put(Attr::Comment("".into())).unwrap();
        a.put(Attr::Places(vec![])).unwrap();
        a.put(Attr::Tag(Tag::from([]))).unwrap();
        a.put(Attr::Ablation(vec![])).unwrap();
        assert!(AttrClass::ALL.iter().all(|&c| !a.has(c)));
        a.put(Attr::WavesTaken(vec![])).unwrap();
        assert!(a.has(AttrClass::WavesTaken));
    }

    #[test]
    fn put_replaces_and_drop_removes() {
        let mut a = Attrs::default();
        a.put(Attr::Comment("one".into())).unwrap();
        a.put(Attr::Comment("two".into())).unwrap();
        assert!(matches!(a.get(AttrClass::Comment), Some(Attr::Comment(t)) if &*t == "two"));
        a.put(Attr::Comment("".into())).unwrap();
        assert!(!a.has(AttrClass::Comment));
        a.put(Attr::Ablation(vec!["b".into(), "a".into()])).unwrap();
        assert_eq!(a.ablation, vec![Arc::from("a"), Arc::from("b")]);
        a.drop(AttrClass::Ablation);
        assert!(!a.has(AttrClass::Ablation));
    }

    #[test]
    fn waves_past_63_are_refused() {
        let mut a = Attrs::default();
        a.put(Attr::WavesTaken(vec![0, 3, 63])).unwrap();
        assert_eq!(a.waves_taken, Some(1 << 63 | 0b1001));
        let Err(Error::Value(msg)) = a.put(Attr::WavesTaken(vec![64])) else {
            panic!("wave 64 passed")
        };
        assert_eq!(msg, "WavesTaken: wave 64 (a workgroup has 64 at most)");
    }

    #[test]
    fn noorder_tags_are_noted() {
        let mut a = Attrs::default();
        a.put(Attr::Tag(Tag::from([crate::Atom::Str("noorder".into())])))
            .unwrap();
        assert!(a.noorder);
        a.drop(AttrClass::Tag);
        assert!(!a.noorder);
    }
}
