//! Making values.

use std::sync::Arc;

use super::Builder;
use crate::error::{value, Result};
use crate::types::Ty;
use crate::values::Typed;

/// What `F.<type>(…)` was given.
#[derive(Clone, Debug, Default)]
pub struct Make {
    /// The element count, if an int (not a bool) was given; `n_repr` is
    /// how it was written.
    pub n: Option<i64>,
    /// The count as written.
    pub n_repr: String,
    /// The name.
    pub name: Option<String>,
    /// The pin.
    pub at: Option<u32>,
    /// The value it shares registers with.
    pub same_as: Option<Typed>,
    /// `short(loc)` of the author's line.
    pub where_: String,
}

impl Builder {
    /// `F.<type>(n=, name=, at=, same_as=)` without its init.
    pub fn make_value(&mut self, file: char, asked: Ty, m: Make) -> Result<Typed> {
        let n = match m.n {
            Some(n) if n >= 1 => n as u32,
            _ => {
                return value(format!(
                    "{file}.{}(n={}): one element or more",
                    asked.name(),
                    m.n_repr
                ))
            }
        };
        let ty = if asked == Ty::Mask {
            Ty::mask_of(self.t.definition().wave)
        } else {
            asked
        };
        if ty.is_mask() && file != 's' {
            return value(format!(
                "a mask in the {file} file: a lane mask is SGPRs (S.mask())"
            ));
        }
        if ty == Ty::F32x2 && file != 'v' {
            return value(format!(
                "an f32x2 in the {file} file: a VGPR pair (V.f32x2())"
            ));
        }
        if let Some(name) = &m.name {
            let ok = !name.is_empty()
                && name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.');
            if !ok {
                return value(format!(
                    "name {}: letters, digits, _ and . only",
                    crate::maker::py_repr(name)
                ));
            }
        }
        let group = self.next_name(m.name.as_deref().unwrap_or("t"));
        let width = n * ty.dwords();
        let align = match width {
            w if w >= 4 => 4,
            2 | 3 => 2,
            _ => 1,
        };
        let v = self.group_value(file, &group, width, align)?;
        if let Some(at) = m.at {
            let values = &mut self.values;
            self.files[super::file_index(file)].pin(values, &group, at)?;
        }
        if let Some(y) = &m.same_as {
            let whole = y.file == file && y.sub == 0 && y.width() == self.values[y.value].n;
            if !whole || y.builder != self.id {
                return value(format!(
                    "same_as={}: a whole value of the {file} file",
                    y.repr()
                ));
            }
            let other = self.values[y.value].name.clone();
            let values = &mut self.values;
            self.files[super::file_index(file)].same_as(values, &group, &other)?;
        }
        Ok(Typed {
            builder: self.id,
            file,
            value: v,
            group: group.into(),
            ty,
            n,
            sub: 0,
            label: m.name.map(Into::into),
            where_: m.where_.into(),
            part: "".into(),
            divisor: false,
        })
    }

    /// The group's one value, made or (a loop body's later pass) reused.
    fn group_value(&mut self, file: char, group: &str, width: u32, align: u32) -> Result<usize> {
        let loc = self.loc;
        let f = &self.files[super::file_index(file)];
        if let Some(g) = f.group(group) {
            let w0 = g.values.first().map_or(0, |&v| self.values[v].n);
            if g.values.len() != 1 || w0 != width {
                return value(format!(
                    "{file}:{group}: made again as {width} dword(s), first as {w0} - a loop body's passes must make \
                     the same values"
                ));
            }
            let v = g.values[0];
            self.values[v].loc = loc;
            self.values[v].where_ = self.where_text.as_str().into();
            return Ok(v);
        }
        let values = &mut self.values;
        let g = self.files[super::file_index(file)].alloc(values, group, 1, width, align, loc)?;
        let v = self.files[super::file_index(file)].groups[g].values[0];
        self.values[v].where_ = self.where_text.as_str().into();
        Ok(v)
    }

    /// `F.<type>(init, …)`: the value, then `init` assigned to each
    /// element in turn.
    pub fn make_init(
        &mut self,
        file: char,
        asked: Ty,
        m: Make,
        init: Option<crate::expr::node::Tree>,
    ) -> Result<Typed> {
        let v = self.make_value(file, asked, m)?;
        if let Some(t) = init {
            for j in 0..v.n {
                let e = if v.n == 1 {
                    v.clone()
                } else {
                    v.element(i64::from(j))?
                };
                self.assign(&e, t.clone(), None)?;
            }
        }
        Ok(v)
    }

    /// `F.entry(*groups)`: per group a typed value, u64 for two
    /// dwords, else u32.
    pub fn entry_values(&mut self, file: char, groups: &[(String, u32)]) -> Result<Vec<Typed>> {
        let refs: Vec<(&str, u32)> = groups.iter().map(|(n, w)| (n.as_str(), *w)).collect();
        let (loc, values) = (self.loc, &mut self.values);
        self.files[super::file_index(file)].entry(values, &refs, loc)?;
        let typed = |(name, w): &(String, u32)| {
            let ty = if *w == 2 { Ty::U64 } else { Ty::U32 };
            self.typed_of(file, name, ty, Some(name))
        };
        Ok(groups.iter().map(typed).collect())
    }

    /// The register an `f32x2(x)` cast checks: x's group's first value's
    /// pin, or 0, plus x's `sub`.
    pub fn pair_reg(&self, x: &Typed) -> u32 {
        self.values.get(x.value).and_then(|v| v.pin).unwrap_or(0) + x.sub
    }

    /// A temporary: one element of its file, base `t`; u64 for a
    /// wide request, the wave's mask for a mask, u32 for cond or none.
    pub fn temp(&mut self, file: char, ty: Option<Ty>) -> Result<Typed> {
        let ty = match ty {
            Some(t) if t.is_wide() => Ty::U64,
            Some(t @ (Ty::Mask | Ty::Mask32 | Ty::F32x2 | Ty::U32 | Ty::I32 | Ty::F32)) => t,
            _ => Ty::U32,
        };
        let file = if ty.is_mask() { 's' } else { file };
        let where_ = self.where_text.clone();
        self.make_value(
            file,
            ty,
            Make {
                n: Some(1),
                n_repr: "1".into(),
                where_,
                ..Make::default()
            },
        )
    }

    /// A typed value of a group's first value, as the entry and the
    /// argument loads make them.
    pub(crate) fn typed_of(&self, file: char, group: &str, ty: Ty, label: Option<&str>) -> Typed {
        let g = self.file(file).group(group).expect("a group");
        let v = g.values[0];
        Typed {
            builder: self.id,
            file,
            value: v,
            group: Arc::from(group),
            ty,
            n: self.values[v].n / ty.dwords().max(1),
            sub: 0,
            label: label.map(Into::into),
            where_: self.where_text.as_str().into(),
            part: "".into(),
            divisor: false,
        }
    }
}
