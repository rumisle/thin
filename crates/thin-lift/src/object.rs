//! Reading a code object with any number of kernels: the inverse of thin's
//! code-object writer (writes one kernel; a vendor library's code object
//! can hold hundreds). An ELF64 AMDGPU image gives its
//! sections, its symbols and, per kernel, its code bytes, its 64-byte
//! descriptor and its metadata map.

use thin_formats::{msgpack_read, Value};

/// A refusal to read an image, with why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObjectError(pub String);

impl std::fmt::Display for ObjectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ObjectError {}

/// A section header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Section {
    /// Its name.
    pub name: String,
    /// sh_type.
    pub kind: u32,
    /// sh_flags.
    pub flags: u64,
    /// sh_addr.
    pub addr: u64,
    /// sh_offset.
    pub offset: u64,
    /// sh_size.
    pub size: u64,
}

/// A kernel of a code object.
#[derive(Clone, Debug, PartialEq)]
pub struct Kernel {
    /// Its symbol.
    pub name: String,
    /// Its code's virtual address.
    pub addr: u64,
    /// Its code's byte range in the image (alignment padding after the last
    /// instruction left out for an unsized symbol).
    pub code: std::ops::Range<usize>,
    /// Its kernel descriptor (the `<name>.kd` symbol), when present.
    pub descriptor: Option<[u8; 64]>,
    /// Its entry in the metadata's `amdhsa.kernels`, when present.
    pub metadata: Option<Value>,
}

/// A code object's sections, kernels and metadata note.
#[derive(Clone, Debug, PartialEq)]
pub struct Object {
    /// e_flags.
    pub e_flags: u32,
    /// The target its machine number names.
    pub target: &'static str,
    /// Every section in header order.
    pub sections: Vec<Section>,
    /// The kernels, by code address.
    pub kernels: Vec<Kernel>,
    /// The first metadata note, when present (the single-kernel writer's map).
    pub metadata: Option<Value>,
    /// Every metadata note in section/note order, including extension fields.
    pub metadata_notes: Vec<Value>,
}

const SHT_SYMTAB: u32 = 2;
const SHT_NOTE: u32 = 7;
const SHF_EXECINSTR: u64 = 4;
const STT_OBJECT: u8 = 1;
const STT_FUNC: u8 = 2;
const NT_AMDGPU_METADATA: u32 = 32;
/// `s_code_end`, the padding some toolchains (e.g. Tensile) put between
/// kernels.
const S_CODE_END: u32 = 0xBF9F_0000;

struct Image<'a>(&'a [u8]);

impl Image<'_> {
    fn bytes(&self, at: usize, n: usize, what: &str) -> Result<&[u8], ObjectError> {
        (self.0.get(at..at.saturating_add(n)))
            .ok_or_else(|| ObjectError(format!("{what} at {at:#x} past the image's end")))
    }
    fn u16(&self, at: usize, what: &str) -> Result<u16, ObjectError> {
        Ok(u16::from_le_bytes(
            self.bytes(at, 2, what)?.try_into().expect("2 bytes"),
        ))
    }
    fn u32(&self, at: usize, what: &str) -> Result<u32, ObjectError> {
        Ok(u32::from_le_bytes(
            self.bytes(at, 4, what)?.try_into().expect("4 bytes"),
        ))
    }
    fn u64(&self, at: usize, what: &str) -> Result<u64, ObjectError> {
        Ok(u64::from_le_bytes(
            self.bytes(at, 8, what)?.try_into().expect("8 bytes"),
        ))
    }
    fn cstr(&self, at: usize) -> String {
        let rest = self.0.get(at..).unwrap_or(&[]);
        let end = rest.iter().position(|&c| c == 0).unwrap_or(rest.len());
        String::from_utf8_lossy(&rest[..end]).into_owned()
    }
}

/// A symbol: (name, info, section index, value, size).
type Symbol = (String, u8, u16, u64, u64);

fn symbols(
    img: &Image<'_>,
    sections: &[Section],
    links: &[u32],
) -> Result<Vec<Symbol>, ObjectError> {
    let mut out = Vec::new();
    for (i, s) in sections.iter().enumerate() {
        if s.kind != SHT_SYMTAB {
            continue;
        }
        let strtab = sections
            .get(links[i] as usize)
            .ok_or_else(|| ObjectError("a symbol table's strings section is missing".into()))?
            .offset as usize;
        for at in (s.offset as usize..(s.offset + s.size) as usize).step_by(24) {
            let name = img.cstr(strtab + img.u32(at, "a symbol")? as usize);
            let info = img.bytes(at + 4, 1, "a symbol")?[0];
            let shndx = img.u16(at + 6, "a symbol")?;
            out.push((
                name,
                info,
                shndx,
                img.u64(at + 8, "a symbol")?,
                img.u64(at + 16, "a symbol")?,
            ));
        }
    }
    Ok(out)
}

/// Reads the image's metadata notes. Objects linked by some toolchains
/// (e.g. Tensile) carry one NT_AMDGPU_METADATA note per kernel, not just
/// one per ELF image.
fn metadata(img: &Image<'_>, sections: &[Section]) -> Result<Vec<Value>, ObjectError> {
    let mut notes = Vec::new();
    for s in sections.iter().filter(|s| s.kind == SHT_NOTE) {
        let (mut at, end) = (s.offset as usize, (s.offset + s.size) as usize);
        while at + 12 <= end {
            let (namesz, descsz) = (
                img.u32(at, "a note")? as usize,
                img.u32(at + 4, "a note")? as usize,
            );
            let kind = img.u32(at + 8, "a note")?;
            let desc = at + 12 + namesz.div_ceil(4) * 4;
            if kind == NT_AMDGPU_METADATA && img.cstr(at + 12) == "AMDGPU" {
                let bytes = img.bytes(desc, descsz, "the metadata note")?;
                let (value, _) = msgpack_read(bytes).map_err(|e| ObjectError(e.to_string()))?;
                notes.push(value);
            }
            at = desc + descsz.div_ceil(4) * 4;
        }
    }
    Ok(notes)
}

fn str_of<'v>(v: &'v Value, key: &str) -> Option<&'v str> {
    match v.get(key) {
        Some(Value::Str(s)) => Some(s),
        _ => None,
    }
}

/// Reads an ELF64 AMDGPU image.
pub fn read(img: &[u8]) -> Result<Object, ObjectError> {
    if img.len() < 64 || &img[..4] != b"\x7fELF" || img[4] != 2 || img[5] != 1 {
        return Err(ObjectError("not a little-endian ELF64".into()));
    }
    let image = Image(img);
    let e_flags = image.u32(0x30, "e_flags")?;
    let target = thin_formats::target_of(e_flags, &thin_targets::MACHINES)
        .map_err(|e| ObjectError(e.to_string()))?;
    let shoff = image.u64(0x28, "e_shoff")? as usize;
    let (shentsize, shnum) = (
        image.u16(0x3A, "e_shentsize")? as usize,
        image.u16(0x3C, "e_shnum")? as usize,
    );
    let shstrndx = image.u16(0x3E, "e_shstrndx")? as usize;
    let mut raw = Vec::new();
    for i in 0..shnum {
        let h = shoff + i * shentsize;
        let s = Section {
            name: String::new(),
            kind: image.u32(h + 4, "a section header")?,
            flags: image.u64(h + 8, "a section header")?,
            addr: image.u64(h + 0x10, "a section header")?,
            offset: image.u64(h + 0x18, "a section header")?,
            size: image.u64(h + 0x20, "a section header")?,
        };
        raw.push((
            image.u32(h, "a section header")?,
            s,
            image.u32(h + 0x28, "a section header")?,
        ));
    }
    let names_at = raw.get(shstrndx).map_or(0, |r| r.1.offset as usize);
    let links: Vec<u32> = raw.iter().map(|r| r.2).collect();
    let sections: Vec<Section> = raw
        .into_iter()
        .map(|(name, mut s, _)| {
            s.name = image.cstr(names_at + name as usize);
            s
        })
        .collect();
    let syms = symbols(&image, &sections, &links)?;
    let notes = metadata(&image, &sections)?;
    let meta = notes.first().cloned();
    let section = |i: u16| sections.get(i as usize);
    let mut kernels: Vec<Kernel> = Vec::new();
    for (name, info, shndx, value, size) in &syms {
        let Some(s) = section(*shndx).filter(|s| s.flags & SHF_EXECINSTR != 0) else {
            continue;
        };
        if info & 0xF != STT_FUNC || kernels.iter().any(|k| k.addr == *value) {
            continue;
        }
        let start = (s.offset + (value - s.addr)) as usize;
        kernels.push(Kernel {
            name: name.clone(),
            addr: *value,
            code: start..start + *size as usize,
            descriptor: None,
            metadata: None,
        });
    }
    kernels.sort_by_key(|k| k.addr);
    for i in 0..kernels.len() {
        if !kernels[i].code.is_empty() {
            continue;
        }
        // An unsized symbol (some toolchains, e.g. Tensile, emit them) runs
        // to the next kernel or its
        // section's end, less the padding: zero words and s_code_end.
        let start = kernels[i].code.start;
        let s = sections
            .iter()
            .find(|s| {
                s.flags & SHF_EXECINSTR != 0
                    && (s.offset as usize..(s.offset + s.size) as usize).contains(&start)
            })
            .expect("a kernel lies in its section");
        let end_of_section = (s.offset + s.size) as usize;
        let mut end = kernels
            .get(i + 1)
            .map_or(end_of_section, |k| k.code.start.min(end_of_section));
        while end >= start + 4 && matches!(image.u32(end - 4, "code")?, 0 | S_CODE_END) {
            end -= 4;
        }
        kernels[i].code = start..end;
    }
    let entries: Vec<&Value> = notes
        .iter()
        .flat_map(|m| match m.get("amdhsa.kernels") {
            Some(Value::Array(list)) => list.as_slice(),
            _ => &[],
        })
        .collect();
    // The first descriptor symbol of each name, and the first metadata
    // entry of each.name and each.symbol: one pass each, not one per
    // kernel.
    let mut descriptors = std::collections::HashMap::new();
    for s in syms.iter().filter(|s| s.1 & 0xF == STT_OBJECT) {
        descriptors.entry(s.0.as_str()).or_insert(s);
    }
    let (mut by_name, mut by_symbol) = (
        std::collections::HashMap::new(),
        std::collections::HashMap::new(),
    );
    for (at, e) in entries.iter().enumerate() {
        if let Some(name) = str_of(e, ".name") {
            by_name.entry(name).or_insert(at);
        }
        if let Some(symbol) = str_of(e, ".symbol") {
            by_symbol.entry(symbol).or_insert(at);
        }
    }
    for k in &mut kernels {
        let kd = format!("{}.kd", k.name);
        if let Some((_, _, shndx, value, _)) = descriptors.get(kd.as_str()) {
            if let Some(s) = section(*shndx) {
                let at = (s.offset + (value - s.addr)) as usize;
                k.descriptor = Some(
                    image
                        .bytes(at, 64, "a kernel descriptor")?
                        .try_into()
                        .expect("64 bytes"),
                );
            }
        }
        let first = [by_name.get(k.name.as_str()), by_symbol.get(kd.as_str())]
            .into_iter()
            .flatten()
            .min();
        k.metadata = first.map(|&at| entries[at].clone());
    }
    Ok(Object {
        e_flags,
        target,
        sections,
        kernels,
        metadata: meta,
        metadata_notes: notes,
    })
}
