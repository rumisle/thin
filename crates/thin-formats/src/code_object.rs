//! The code object: an AMDHSA ELF64 holding one kernel, read, normalized
//! and written with no linker, and the registry
//! of machine numbers.

use sha2::{Digest, Sha256};

use crate::msgpack::{canonical_json, msgpack_read, msgpack_write, Value};
use crate::{Error, Result};

/// EM_AMDGPU.
const EM_AMDGPU: u16 = 224;
/// SHF_ALLOC.
const SHF_ALLOC: u64 = 2;
/// The AMDGPU metadata note's type.
const NT_AMDGPU_METADATA: u32 = 32;
/// The bytes of a kernel descriptor.
pub const DESCRIPTOR_BYTES: usize = 64;

fn refuse(why: String) -> Error {
    Error::CodeObject(why)
}

/// A target's ELF machine number, `e_flags & 0xFF` (EF_AMDGPU_MACH).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Machine {
    /// The machine number.
    pub number: u32,
    /// The target's name.
    pub target: &'static str,
}

/// The target the e_flags select among the targets built in; an unknown
/// machine number is a `ValueError` listing the known ones.
pub fn target_of(e_flags: u32, machines: &[Machine]) -> Result<&'static str> {
    let number = e_flags & 0xFF;
    machines
        .iter()
        .find(|m| m.number == number)
        .map(|m| m.target)
        .ok_or_else(|| {
            let known: Vec<_> = machines
                .iter()
                .map(|m| format!("{:#04x} {}", m.number, m.target))
                .collect();
            Error::Value(format!(
                "e_flags {e_flags:#x}: machine number {number:#04x} is not a known target (known: {})",
                known.join(", ")
            ))
        })
}

/// A section header, as read.
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
    /// sh_link.
    pub link: u32,
}

/// A code object's parts.
#[derive(Clone, Debug, PartialEq)]
pub struct CodeObject {
    /// The OS ABI and its version, `img[7..9]`.
    pub ident: [u8; 2],
    /// e_flags.
    pub e_flags: u32,
    /// Every section in header order, the null one included.
    pub sections: Vec<Section>,
    /// The kernel's symbol: the `.kd` symbol's name without `.kd`.
    pub symbol: String,
    /// The kernel descriptor.
    pub descriptor: [u8; DESCRIPTOR_BYTES],
    /// The decoded metadata note.
    pub metadata: Value,
    /// The bytes of `.text`.
    pub text: Vec<u8>,
}

impl CodeObject {
    fn descriptor_u32(&self, at: usize) -> u32 {
        u32::from_le_bytes(self.descriptor[at..at + 4].try_into().expect("4 bytes"))
    }

    /// The LDS bytes the descriptor asks for (u32 at 0).
    pub fn lds(&self) -> u32 {
        self.descriptor_u32(0)
    }

    /// The private bytes per lane (u32 at 4).
    pub fn private(&self) -> u32 {
        self.descriptor_u32(4)
    }

    /// The kernarg bytes (u32 at 8).
    pub fn kernarg(&self) -> u32 {
        self.descriptor_u32(8)
    }

    /// The target the e_flags select among the targets built in.
    pub fn target(&self, machines: &[Machine]) -> Result<&'static str> {
        target_of(self.e_flags, machines)
    }

    /// The section of that name, if any.
    fn section(&self, name: &str) -> Option<&Section> {
        self.sections.iter().find(|s| s.name == name)
    }
}

/// A bounds-checked view of an image whose every read names what it reads.
struct Image<'a>(&'a [u8]);

impl<'a> Image<'a> {
    fn bytes(&self, at: u64, len: u64, what: &str) -> Result<&'a [u8]> {
        let end = at
            .checked_add(len)
            .filter(|&end| end <= self.0.len() as u64);
        let end = end.ok_or_else(|| {
            refuse(format!(
                "{what}: {len} bytes at {at:#x} run past the {}-byte image",
                self.0.len()
            ))
        })?;
        Ok(&self.0[at as usize..end as usize])
    }

    fn u16(&self, at: u64, what: &str) -> Result<u16> {
        Ok(u16::from_le_bytes(
            self.bytes(at, 2, what)?.try_into().expect("2 bytes"),
        ))
    }

    fn u32(&self, at: u64, what: &str) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.bytes(at, 4, what)?.try_into().expect("4 bytes"),
        ))
    }

    fn u64(&self, at: u64, what: &str) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.bytes(at, 8, what)?.try_into().expect("8 bytes"),
        ))
    }

    /// The NUL-terminated name at `at`.
    fn name(&self, at: u64, what: &str) -> Result<String> {
        let rest = self.bytes(at, 0, what).map(|_| &self.0[at as usize..])?;
        let len = rest
            .iter()
            .position(|&b| b == 0)
            .ok_or_else(|| refuse(format!("{what} at {at:#x} runs past the image")))?;
        String::from_utf8(rest[..len].to_vec())
            .map_err(|_| refuse(format!("{what} at {at:#x} is not UTF-8")))
    }
}

/// The section headers, names resolved.
fn read_sections(img: &Image<'_>) -> Result<Vec<Section>> {
    if img.u16(0x3A, "e_shentsize")? != 64 {
        return Err(refuse("the section header entry size is not 64".into()));
    }
    let shoff = img.u64(0x28, "e_shoff")?;
    let shnum = u64::from(img.u16(0x3C, "e_shnum")?);
    let shstrndx = u64::from(img.u16(0x3E, "e_shstrndx")?);
    img.bytes(shoff, shnum * 64, "the section header table")?;
    if shstrndx >= shnum {
        return Err(refuse(format!(
            "e_shstrndx {shstrndx} is not below e_shnum {shnum}"
        )));
    }
    let names_at = img.u64(shoff + shstrndx * 64 + 24, "the .shstrtab offset")?;
    (0..shnum)
        .map(|i| {
            let h = shoff + i * 64;
            let what = format!("section {i}");
            let name_at = names_at + u64::from(img.u32(h, &what)?);
            Ok(Section {
                name: img.name(name_at, &format!("the name of section {i}"))?,
                kind: img.u32(h + 4, &what)?,
                flags: img.u64(h + 8, &what)?,
                addr: img.u64(h + 16, &what)?,
                offset: img.u64(h + 24, &what)?,
                size: img.u64(h + 32, &what)?,
                link: img.u32(h + 40, &what)?,
            })
        })
        .collect()
}

/// A symbol: its name, section index and value.
struct Symbol {
    name: String,
    shndx: u16,
    value: u64,
}

fn read_symbols(img: &Image<'_>, sections: &[Section]) -> Result<Vec<Symbol>> {
    let dynsym = sections
        .iter()
        .find(|s| s.name == ".dynsym")
        .ok_or_else(|| refuse("no .dynsym section".into()))?;
    let strtab = sections.get(dynsym.link as usize).ok_or_else(|| {
        refuse(format!(
            ".dynsym's sh_link {} names no section",
            dynsym.link
        ))
    })?;
    let table = img.bytes(dynsym.offset, dynsym.size, ".dynsym")?;
    (0..table.len() as u64 / 24)
        .map(|i| {
            let at = dynsym.offset + i * 24;
            let what = format!("symbol {i}");
            let shndx = img.u16(at + 6, &what)?;
            if usize::from(shndx) >= sections.len() {
                return Err(refuse(format!(
                    "symbol {i}'s section index {shndx} is out of range"
                )));
            }
            let name_at = strtab.offset + u64::from(img.u32(at, &what)?);
            Ok(Symbol {
                name: img.name(name_at, &format!("the name of symbol {i}"))?,
                shndx,
                value: img.u64(at + 8, &what)?,
            })
        })
        .collect()
}

/// The AMDGPU metadata note's msgpack bytes.
fn read_note<'a>(img: &Image<'a>, note: &Section) -> Result<&'a [u8]> {
    let data = img.bytes(note.offset, note.size, ".note")?;
    let mut at = 0u64;
    let pad4 = |n: u64| n.div_ceil(4) * 4;
    while at < data.len() as u64 {
        let note_img = Image(data);
        let truncated = |_| refuse(format!("the note at {at:#x} of .note is truncated"));
        let namesz = u64::from(note_img.u32(at, "a note").map_err(truncated)?);
        let descsz = u64::from(note_img.u32(at + 4, "a note").map_err(truncated)?);
        let kind = note_img.u32(at + 8, "a note").map_err(truncated)?;
        let name = note_img
            .bytes(at + 12, namesz, "a note")
            .map_err(truncated)?;
        let desc_at = at + 12 + pad4(namesz);
        let desc = note_img
            .bytes(desc_at, descsz, "a note")
            .map_err(truncated)?;
        let owner = name
            .iter()
            .rposition(|&b| b != 0)
            .map_or(&name[..0], |end| &name[..=end]);
        if owner == b"AMDGPU" && kind == NT_AMDGPU_METADATA {
            return Ok(desc);
        }
        at = desc_at + pad4(descsz);
    }
    Err(refuse("no AMDGPU note of type 32 in .note".into()))
}

/// Read and validate a code object's sections, descriptor and metadata.
pub fn read(img: &[u8]) -> Result<CodeObject> {
    if img.len() < 64 || &img[..4] != b"\x7fELF" || img[4] != 2 || img[5] != 1 {
        return Err(refuse("not a little-endian ELF64".into()));
    }
    let image = Image(img);
    let machine = image.u16(0x12, "e_machine")?;
    if machine != EM_AMDGPU {
        return Err(refuse(format!(
            "e_machine {machine}, not AMDGPU ({EM_AMDGPU})"
        )));
    }
    let sections = read_sections(&image)?;
    let symbols = read_symbols(&image, &sections)?;
    let mut kd = symbols.iter().filter(|s| s.name.ends_with(".kd"));
    let (Some(kd), None) = (kd.next(), kd.next()) else {
        return Err(refuse("not exactly one symbol ending in .kd".into()));
    };
    let find = |name: &str| {
        sections
            .iter()
            .find(|s| s.name == name)
            .ok_or_else(|| refuse(format!("no {name} section")))
    };
    let text = find(".text")?;
    let note = find(".note")?;
    let desc = read_note(&image, note)?;
    let (metadata, _) = msgpack_read(desc)
        .map_err(|e| refuse(format!("the AMDGPU note is not valid msgpack ({e})")))?;
    let home = &sections[usize::from(kd.shndx)];
    let at = (home.offset.wrapping_add(kd.value)).wrapping_sub(home.addr);
    let descriptor = image.bytes(at, DESCRIPTOR_BYTES as u64, "the kernel descriptor")?;
    Ok(CodeObject {
        ident: [img[7], img[8]],
        e_flags: image.u32(0x30, "e_flags")?,
        symbol: kd.name.trim_end_matches(".kd").to_owned(),
        descriptor: descriptor.try_into().expect("64 bytes"),
        metadata,
        text: image.bytes(text.offset, text.size, ".text")?.to_vec(),
        sections,
    })
}

/// The normalized "what runs" of a code object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Code {
    /// The OS ABI and its version.
    pub ident: [u8; 2],
    /// e_flags.
    pub e_flags: u32,
    /// The bytes of `.text`.
    pub text: Vec<u8>,
    /// The descriptor with its entry offset (bytes 16–23) zeroed.
    pub descriptor: [u8; DESCRIPTOR_BYTES],
    /// The metadata without the kernel's `.name` and `.symbol`, as
    /// canonical JSON.
    pub metadata: String,
}

impl Code {
    /// The names of [`digests`](Self::digests), in order.
    pub const DIGESTS: [&'static str; 5] = ["ident", "e_flags", "text", "descriptor", "metadata"];

    /// The sha256 hex digests of ident, e_flags (u32 LE), text,
    /// descriptor and metadata (UTF-8), in the order of [`Code::DIGESTS`].
    pub fn digests(&self) -> [String; 5] {
        let hex = |bytes: &[u8]| format!("{:x}", Sha256::digest(bytes));
        [
            hex(&self.ident),
            hex(&self.e_flags.to_le_bytes()),
            hex(&self.text),
            hex(&self.descriptor),
            hex(self.metadata.as_bytes()),
        ]
    }
}

/// The sections a loaded code object may hold.
const ALLOCATED: [&str; 9] = [
    ".note",
    ".dynsym",
    ".gnu.hash",
    ".hash",
    ".dynstr",
    ".rodata",
    ".text",
    ".dynamic",
    ".relro_padding",
];

/// The metadata's one kernel map, refusing anything but exactly one whose
/// `.name` and `.symbol` are the symbol's.
fn kernel_map<'v>(metadata: &'v Value, symbol: &str) -> Result<&'v [(String, Value)]> {
    let kernels = match metadata.get("amdhsa.kernels") {
        Some(Value::Array(kernels)) => kernels,
        _ => return Err(refuse("the metadata's amdhsa.kernels is not a list".into())),
    };
    let [Value::Map(kernel)] = kernels.as_slice() else {
        return Err(refuse(format!(
            "the metadata's amdhsa.kernels holds {} values, not one map",
            kernels.len()
        )));
    };
    let field = |key: &str| match kernel.iter().find(|(k, _)| k == key) {
        Some((_, Value::Str(s))) => Some(s.as_str()),
        _ => None,
    };
    let kd = format!("{symbol}.kd");
    if (field(".name"), field(".symbol")) != (Some(symbol), Some(kd.as_str())) {
        return Err(refuse(format!(
            "the kernel's .name and .symbol are not ({symbol:?}, {kd:?})"
        )));
    }
    Ok(kernel)
}

/// The normalized code of a code object, refusing what refuses.
pub fn code(img: &[u8]) -> Result<Code> {
    let object = read(img)?;
    if let Some(s) = (object.sections.iter())
        .find(|s| s.flags & SHF_ALLOC != 0 && !ALLOCATED.contains(&s.name.as_str()))
    {
        return Err(refuse(format!(
            "the allocated section {:?} is not one a kernel holds",
            s.name
        )));
    }
    let rodata = object
        .section(".rodata")
        .filter(|s| s.size == DESCRIPTOR_BYTES as u64)
        .ok_or_else(|| refuse("no .rodata of exactly 64 bytes".into()))?;
    let text = object.section(".text").expect("read found .text");
    let entry = i64::from_le_bytes(object.descriptor[16..24].try_into().expect("8 bytes"));
    if rodata.addr.wrapping_add_signed(entry) != text.addr {
        return Err(refuse(format!(
            "the descriptor's entry {entry} from .rodata at {:#x} is not .text at {:#x}",
            rodata.addr, text.addr
        )));
    }
    let kernel = kernel_map(&object.metadata, &object.symbol)?;
    let kept: Vec<(String, Value)> = (kernel.iter())
        .filter(|(k, _)| k != ".name" && k != ".symbol")
        .cloned()
        .collect();
    let mut metadata = object.metadata.clone();
    if let Value::Map(entries) = &mut metadata {
        for (key, value) in entries.iter_mut() {
            if key == "amdhsa.kernels" {
                *value = Value::Array(vec![Value::Map(kept.clone())]);
            }
        }
    }
    let mut descriptor = object.descriptor;
    descriptor[16..24].fill(0);
    Ok(Code {
        ident: object.ident,
        e_flags: object.e_flags,
        text: object.text,
        descriptor,
        metadata: canonical_json(&metadata),
    })
}

/// The GNU hash of a symbol name.
fn gnu_hash(name: &str) -> u32 {
    name.bytes().fold(5381u32, |h, b| {
        h.wrapping_mul(33).wrapping_add(u32::from(b))
    })
}

const fn align(n: u64, to: u64) -> u64 {
    n.div_ceil(to) * to
}

/// One section the writer lays out.
struct Out {
    name: &'static str,
    kind: u32,
    flags: u64,
    link: u32,
    info: u32,
    align: u64,
    entsize: u64,
    bytes: Vec<u8>,
    offset: u64,
    addr: u64,
}

impl Out {
    fn new(
        name: &'static str,
        kind: u32,
        flags: u64,
        link: u32,
        align: u64,
        entsize: u64,
        bytes: Vec<u8>,
    ) -> Out {
        let info = u32::from(matches!(name, ".dynsym" | ".symtab"));
        Out {
            name,
            kind,
            flags,
            link,
            info,
            align,
            entsize,
            bytes,
            offset: 0,
            addr: 0,
        }
    }
}

fn le(fields: &[(u64, usize)]) -> Vec<u8> {
    fields
        .iter()
        .flat_map(|&(value, width)| value.to_le_bytes()[..width].to_vec())
        .collect()
}

/// Checks the arguments of [`write`] against refusals.
fn check_write(
    symbol: &str,
    text: &[u8],
    descriptor: &[u8],
    metadata: &Value,
    e_flags: u32,
    machines: &[Machine],
) -> Result<()> {
    if symbol.is_empty() || symbol.contains('\0') {
        return Err(refuse(format!(
            "the symbol {symbol:?} is empty or holds a NUL"
        )));
    }
    if descriptor.len() != DESCRIPTOR_BYTES {
        return Err(refuse(format!(
            "the descriptor is {} bytes, not 64",
            descriptor.len()
        )));
    }
    if text.is_empty() {
        return Err(refuse("the text is empty".into()));
    }
    target_of(e_flags, machines).map_err(|e| refuse(e.to_string()))?;
    kernel_map(metadata, symbol).map(|_| ())
}

/// The code object of one kernel, laid out as places it, with no
/// linker; `machines` are the targets built in, which the e_flags must name.
pub fn write(
    symbol: &str,
    text: &[u8],
    descriptor: &[u8],
    metadata: &Value,
    e_flags: u32,
    machines: &[Machine],
) -> Result<Vec<u8>> {
    check_write(symbol, text, descriptor, metadata, e_flags, machines)?;
    let kd = format!("{symbol}.kd");
    let mut desc = msgpack_write(metadata)?;
    let descsz = desc.len() as u64;
    desc.resize(align(descsz, 4) as usize, 0);
    let mut note = le(&[(7, 4), (descsz, 4), (u64::from(NT_AMDGPU_METADATA), 4)]);
    note.extend(b"AMDGPU\0\0");
    note.extend(&desc);
    let dynstr = format!("\0{symbol}\0{kd}\0").into_bytes();
    let (h_sym, h_kd) = (gnu_hash(symbol), gnu_hash(&kd));
    let bloom = [h_sym, h_kd].iter().fold(0u64, |bloom, &h| {
        bloom | 1 << (h % 64) | 1 << ((h >> 26) % 64)
    });
    let gnu = le(&[
        (1, 4),
        (1, 4),
        (1, 4),
        (26, 4),
        (bloom, 8),
        (1, 4),
        (u64::from(h_sym & !1), 4),
        (u64::from(h_kd | 1), 4),
    ]);
    let hash = le(&[(1, 4), (3, 4), (2, 4), (0, 4), (0, 4), (1, 4)]);
    let mut sections = vec![
        Out::new(".note", 7, 2, 0, 4, 0, note),
        Out::new(".dynsym", 11, 2, 5, 8, 24, vec![0; 72]),
        Out::new(".gnu.hash", 0x6FFF_FFF6, 2, 2, 8, 0, gnu),
        Out::new(".hash", 5, 2, 2, 4, 4, hash),
        Out::new(".dynstr", 3, 2, 0, 1, 0, dynstr.clone()),
        Out::new(".rodata", 1, 2, 0, 64, 0, descriptor.to_vec()),
        Out::new(".text", 1, 6, 0, 256, 0, text.to_vec()),
        Out::new(".dynamic", 6, 3, 5, 8, 16, vec![0; 112]),
    ];
    let mut at = 64 + 6 * 56;
    for s in &mut sections[..6] {
        at = align(at, s.align);
        (s.offset, s.addr) = (at, at);
        at += s.bytes.len() as u64;
    }
    let r_end = at;
    let text_offset = align(r_end, 256);
    let text_addr = align(r_end, 4096) + text_offset % 4096;
    let dynamic_offset = align(text_offset + text.len() as u64, 8);
    let dynamic_addr = align(text_addr + text.len() as u64, 4096) + dynamic_offset % 4096;
    (sections[6].offset, sections[6].addr) = (text_offset, text_addr);
    (sections[7].offset, sections[7].addr) = (dynamic_offset, dynamic_addr);
    let addr = |i: usize| sections[i].addr;
    let rodata_addr = addr(5);
    let dynsym = [
        vec![0; 24],
        le(&[
            (1, 4),
            (0x12, 1),
            (3, 1),
            (7, 2),
            (text_addr, 8),
            (text.len() as u64, 8),
        ]),
        le(&[
            (2 + symbol.len() as u64, 4),
            (0x11, 1),
            (3, 1),
            (6, 2),
            (rodata_addr, 8),
            (64, 8),
        ]),
    ]
    .concat();
    let dynamic = le(&[
        (6, 8),
        (addr(1), 8),
        (11, 8),
        (24, 8),
        (5, 8),
        (addr(4), 8),
        (10, 8),
        (dynstr.len() as u64, 8),
        (0x6FFF_FEF5, 8),
        (addr(2), 8),
        (4, 8),
        (addr(3), 8),
        (0, 8),
        (0, 8),
    ]);
    sections[1].bytes = dynsym.clone();
    sections[5].bytes[16..24].copy_from_slice(&(text_addr.wrapping_sub(rodata_addr)).to_le_bytes());
    sections[7].bytes = dynamic;
    sections.push(Out::new(".symtab", 2, 0, 10, 8, 24, dynsym));
    sections.push(Out::new(".strtab", 3, 0, 0, 1, 0, dynstr));
    let mut shstrtab = vec![0u8];
    let mut names = Vec::new();
    for name in sections.iter().map(|s| s.name).chain([".shstrtab"]) {
        names.push(shstrtab.len() as u64);
        shstrtab.extend(name.bytes().chain([0]));
    }
    sections.push(Out::new(".shstrtab", 3, 0, 0, 1, 0, shstrtab));
    let mut at = dynamic_offset + 112;
    for s in &mut sections[8..] {
        at = align(at, s.align);
        s.offset = at;
        at += s.bytes.len() as u64;
    }
    let shoff = align(at, 8);
    let mut img = vec![0u8; (shoff + 12 * 64) as usize];
    for s in &sections {
        img[s.offset as usize..s.offset as usize + s.bytes.len()].copy_from_slice(&s.bytes);
    }
    let header = [
        b"\x7fELF\x02\x01\x01\x40\x04".to_vec(),
        vec![0; 7],
        le(&[
            (3, 2),
            (u64::from(EM_AMDGPU), 2),
            (1, 4),
            (0, 8),
            (64, 8),
            (shoff, 8),
            (u64::from(e_flags), 4),
        ]),
        le(&[(64, 2), (56, 2), (6, 2), (64, 2), (12, 2), (11, 2)]),
    ]
    .concat();
    img[..64].copy_from_slice(&header);
    let (note, text_len) = (&sections[0], text.len() as u64);
    let segments = [
        (6, 4, 64, 64, 336, 8),
        (1, 4, 0, 0, r_end, 4096),
        (1, 5, text_offset, text_addr, text_len, 4096),
        (1, 6, dynamic_offset, dynamic_addr, 112, 4096),
        (2, 6, dynamic_offset, dynamic_addr, 112, 8),
        (4, 4, note.offset, note.addr, note.bytes.len() as u64, 4),
    ];
    for (i, &(kind, flags, offset, vaddr, size, align)) in segments.iter().enumerate() {
        let at = 64 + i * 56;
        let phdr = le(&[
            (kind, 4),
            (flags, 4),
            (offset, 8),
            (vaddr, 8),
            (vaddr, 8),
            (size, 8),
            (size, 8),
            (align, 8),
        ]);
        img[at..at + 56].copy_from_slice(&phdr);
    }
    for (i, s) in sections.iter().enumerate() {
        let at = (shoff + (i as u64 + 1) * 64) as usize;
        let shdr = le(&[
            (names[i], 4),
            (u64::from(s.kind), 4),
            (s.flags, 8),
            (s.addr, 8),
            (s.offset, 8),
            (s.bytes.len() as u64, 8),
            (u64::from(s.link), 4),
            (u64::from(s.info), 4),
            (s.align, 8),
            (s.entsize, 8),
        ]);
        img[at..at + 64].copy_from_slice(&shdr);
    }
    Ok(img)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The vector image, written for symbol `k` with e_flags 0x4E.
    const IMAGE: &[u8] = include_bytes!("../tests/data/formats/codeobject_k.co");
    const MACHINES: [Machine; 1] = [Machine {
        number: 0x4E,
        target: "test",
    }];

    fn patched(at: usize, bytes: &[u8]) -> Vec<u8> {
        let mut img = IMAGE.to_vec();
        img[at..at + bytes.len()].copy_from_slice(bytes);
        img
    }

    /// The file offset of section `i`'s header.
    fn header(i: usize) -> usize {
        let shoff = u64::from_le_bytes(IMAGE[0x28..0x30].try_into().unwrap());
        shoff as usize + i * 64
    }

    fn refused(img: &[u8], what: &str) -> bool {
        matches!(read(img), Err(Error::CodeObject(m)) if m.contains(what))
    }

    #[test]
    fn the_vector_reads_and_normalizes() {
        let object = read(IMAGE).unwrap();
        assert_eq!(
            (object.symbol.as_str(), object.lds(), object.kernarg()),
            ("k", 1024, 16)
        );
        assert_eq!(object.private(), 0);
        assert_eq!(object.target(&MACHINES).unwrap(), "test");
        assert!(code(IMAGE).is_ok());
    }

    #[test]
    fn read_refuses_each_malformation() {
        let elf = "not a little-endian ELF64";
        for img in [
            &IMAGE[..63],
            &patched(0, b"\x7fELG"),
            &patched(4, &[1]),
            &patched(5, &[2]),
        ] {
            assert!(matches!(read(img), Err(Error::CodeObject(m)) if m == elf));
        }
        let machine = "e_machine 62, not AMDGPU (224)";
        assert!(
            matches!(read(&patched(0x12, &[0x3E, 0])), Err(Error::CodeObject(m)) if m == machine)
        );
        assert!(refused(&patched(0x3A, &[40, 0]), "entry size"));
        assert!(refused(&patched(0x28, &[0xFF; 4]), "section header table"));
        assert!(refused(&patched(0x3E, &[12, 0]), "e_shstrndx"));
        assert!(refused(
            &patched(header(1), &[0xFF, 0xFF, 0, 0]),
            "name of section 1"
        ));
        let shstrtab = 1736;
        assert!(refused(&patched(shstrtab + 1, &[0xFF]), "not UTF-8"));
        assert!(refused(&patched(shstrtab + 7, b"!"), "no .dynsym"));
        assert!(refused(&patched(header(2) + 40, &[99]), "sh_link"));
        assert!(refused(&patched(624 + 24 + 6, &[99, 0]), "section index"));
        assert!(refused(&patched(756 + 4, b"x"), "exactly one symbol"));
        assert!(refused(&patched(624 + 24, &[3]), "exactly one symbol"));
        assert!(refused(&patched(shstrtab + 48, b"!"), "no .text"));
        assert!(refused(&patched(shstrtab + 1, b"!"), "no .note"));
        assert!(refused(&patched(400 + 12, b"AMDGPV"), "no AMDGPU note"));
        assert!(refused(&patched(400 + 4, &[0xFF, 0xFF]), "truncated"));
        assert!(refused(&patched(400 + 20, &[0xC1]), "not valid msgpack"));
    }

    #[test]
    fn code_refuses_what_runs_differently() {
        let alloc = |i: usize| patched(header(i) + 8, &[2]);
        assert!(matches!(code(&alloc(9)), Err(Error::CodeObject(m)) if m.contains(".symtab")));
        let short = patched(header(6) + 32, &[32]);
        assert!(matches!(code(&short), Err(Error::CodeObject(m)) if m.contains(".rodata")));
        let entry = patched(768 + 16, &[0x10]);
        assert!(matches!(code(&entry), Err(Error::CodeObject(m)) if m.contains("entry")));
        let name = IMAGE.windows(7).position(|w| w == b".name\xa1k").unwrap() + 6;
        let renamed = patched(name, b"j");
        assert!(matches!(code(&renamed), Err(Error::CodeObject(m)) if m.contains(".name")));
    }

    fn metadata(kernels: Vec<Value>) -> Value {
        Value::Map(vec![("amdhsa.kernels".into(), Value::Array(kernels))])
    }

    fn kernel(name: &str, symbol: &str) -> Value {
        Value::Map(vec![
            (".name".into(), Value::Str(name.into())),
            (".symbol".into(), Value::Str(symbol.into())),
        ])
    }

    #[test]
    fn write_refuses_before_writing() {
        let good = metadata(vec![kernel("k", "k.kd")]);
        let write = |symbol, text: &[u8], desc: &[u8], meta: &Value, flags| {
            matches!(
                write(symbol, text, desc, meta, flags, &MACHINES),
                Err(Error::CodeObject(_))
            )
        };
        let desc = [0; 64];
        assert!(!write("k", &[1], &desc, &good, 0x4E));
        assert!(write("", &[1], &desc, &good, 0x4E));
        assert!(write("k\0", &[1], &desc, &good, 0x4E));
        assert!(write("k", &[1], &desc[..63], &good, 0x4E));
        assert!(write("k", &[], &desc, &good, 0x4E));
        assert!(write("k", &[1], &desc, &good, 0x4F));
        assert!(write("k", &[1], &desc, &metadata(vec![]), 0x4E));
        let two = metadata(vec![kernel("k", "k.kd"), kernel("k", "k.kd")]);
        assert!(write("k", &[1], &desc, &two, 0x4E));
        assert!(write(
            "k",
            &[1],
            &desc,
            &metadata(vec![kernel("k", "k")]),
            0x4E
        ));
        let image = super::write("k", &[1], &desc, &good, 0x4E, &MACHINES).unwrap();
        assert_eq!(code(&image).unwrap().metadata, r#"{"amdhsa.kernels":[{}]}"#);
    }

    #[test]
    fn unknown_machines_list_the_known_ones() {
        let error = target_of(0x4F, &MACHINES).unwrap_err().to_string();
        assert!(
            error.contains("0x4f") && error.contains("0x4e test"),
            "{error}"
        );
    }
}
