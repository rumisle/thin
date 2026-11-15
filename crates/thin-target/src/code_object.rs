//! The generic code-object facts: the descriptor block and the kernel
//! descriptor a target's words fit into.

use std::collections::BTreeMap;

use crate::read_s::SFile;
use crate::target::Target;
use crate::Error;

/// The parameters of a kernel's footer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Footer {
    /// The kernel's symbol.
    pub name: String,
    /// The target's triple name (`amdgcn-amd-amdhsa--<target>`).
    pub target_name: String,
    /// LDS bytes.
    pub lds: u32,
    /// Kernel-argument bytes.
    pub kernarg: u32,
    /// Next free VGPR.
    pub vgpr: u32,
    /// Next free AGPR.
    pub agpr: u32,
    /// Next free SGPR.
    pub sgpr: u32,
    /// Work-items per workgroup.
    pub wgsize: u32,
    /// Whether the workgroup id y is enabled.
    pub wgy: u32,
    /// Preloaded argument dwords.
    pub preload: u32,
    /// The body already ends the program: its last instruction is
    /// `s_endpgm` or an unconditional `s_branch`, no label after it, so the
    /// closing `s_endpgm` would be unreachable.
    pub ends: bool,
}

/// A descriptor block's values, by entry name (after `.amdhsa_`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Block(pub BTreeMap<String, u64>);

impl Block {
    /// The value of an entry, refusing a missing one.
    pub fn get(&self, key: &str) -> Result<u64, Error> {
        self.0
            .get(key)
            .copied()
            .ok_or_else(|| Error::Asm(format!(".amdhsa_{key} is missing")))
    }

    /// The value of an entry, refusing one that does not fit `bits` bits.
    pub fn bits(&self, key: &str, bits: u32) -> Result<u64, Error> {
        let value = self.get(key)?;
        if value >> bits != 0 {
            return Err(Error::Asm(format!(
                ".amdhsa_{key} {value}: does not fit its {bits} bits"
            )));
        }
        Ok(value)
    }
}

/// A target's own descriptor words.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Resources {
    /// Its bits of rsrc1.
    pub rsrc1: u32,
    /// rsrc3.
    pub rsrc3: u32,
    /// Its kernel-property bits.
    pub props: u32,
    /// The kernarg-preload word.
    pub preload: u32,
    /// The user SGPRs of preloaded arguments.
    pub user_sgprs: u32,
}

/// The block directives every target takes (without `.amdhsa_`).
pub static COMMON: [&str; 24] = [
    "group_segment_fixed_size",
    "private_segment_fixed_size",
    "kernarg_size",
    "user_sgpr_count",
    "user_sgpr_dispatch_ptr",
    "user_sgpr_queue_ptr",
    "user_sgpr_kernarg_segment_ptr",
    "user_sgpr_dispatch_id",
    "user_sgpr_private_segment_size",
    "uses_dynamic_stack",
    "enable_private_segment",
    "system_sgpr_workgroup_id_x",
    "system_sgpr_workgroup_id_y",
    "system_sgpr_workgroup_id_z",
    "system_sgpr_workgroup_info",
    "system_vgpr_workitem_id",
    "next_free_vgpr",
    "next_free_sgpr",
    "reserve_vcc",
    "float_round_mode_32",
    "float_round_mode_16_64",
    "float_denorm_mode_32",
    "float_denorm_mode_16_64",
    "fp16_overflow",
];

/// A block value: `0`, a decimal without a leading zero, or `0x` hex.
fn block_value(key: &str, text: &str, line: usize) -> Result<u64, Error> {
    let parsed = if let Some(hex) = text.strip_prefix("0x") {
        (!hex.is_empty() && hex.bytes().all(|b| b.is_ascii_hexdigit()))
            .then(|| u64::from_str_radix(hex, 16).ok())
            .flatten()
    } else if text == "0"
        || (!text.starts_with('0') && !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit()))
    {
        text.parse().ok()
    } else {
        None
    };
    parsed.ok_or_else(|| Error::Asm(format!("line {line}: .amdhsa_{key} {text}: not an integer")))
}

/// The 64-byte kernel descriptor of a `.s`, from its block and the
/// target's words.
pub fn descriptor(file: &SFile, target: &Target) -> Result<[u8; 64], Error> {
    let block = descriptor_block(file, target)?;
    let resources = target.hooks().resources(&block)?;
    descriptor_words(&block, target, resources)
}

/// Encodes descriptor interface fields without choosing a register allocation.
/// Only group/private/kernarg sizes, rsrc2 and kernel-property interface bits
/// are meaningful in the result. The full writer's resource-domain checks
/// remain in [`descriptor`]; both entry points share the same bit encoder.
pub fn interface_descriptor(file: &SFile, target: &Target) -> Result<[u8; 64], Error> {
    let block = descriptor_block(file, target)?;
    let resources = target.hooks().interface_resources(&block)?;
    descriptor_words(&block, target, resources)
}

fn descriptor_block(file: &SFile, target: &Target) -> Result<Block, Error> {
    if !file.once.contains_key(".amdhsa_kernel") {
        return Err(Error::Asm("no .amdhsa_kernel".into()));
    }
    if let Some((version, line)) = file.once.get(".amdhsa_code_object_version") {
        if version != "6" {
            return Err(Error::Asm(format!(
                "line {line}: .amdhsa_code_object_version {version}: only 6"
            )));
        }
    }
    let mut block = Block::default();
    for (key, (text, line)) in &file.block {
        block.0.insert(key.clone(), block_value(key, text, *line)?);
    }
    let hooks = target.hooks();
    let mut wanted: Vec<&str> = COMMON.iter().chain(hooks.directives()).copied().collect();
    wanted.sort_unstable();
    let given: Vec<&str> = block.0.keys().map(String::as_str).collect();
    if given != wanted {
        let missing: Vec<_> = wanted.iter().filter(|k| !given.contains(k)).collect();
        let extra: Vec<_> = given.iter().filter(|k| !wanted.contains(k)).collect();
        return Err(Error::Asm(format!(
            "the descriptor block's directives: missing {missing:?}, not taken {extra:?}"
        )));
    }
    Ok(block)
}

fn descriptor_words(block: &Block, target: &Target, res: Resources) -> Result<[u8; 64], Error> {
    let flag = |key: &str| block.bits(key, 1).map(|v| v as u32);
    let (dispatch, queue, kernarg_ptr, dispatch_id) = (
        flag("user_sgpr_dispatch_ptr")?,
        flag("user_sgpr_queue_ptr")?,
        flag("user_sgpr_kernarg_segment_ptr")?,
        flag("user_sgpr_dispatch_id")?,
    );
    let private_size = flag("user_sgpr_private_segment_size")?;
    let user = block.bits("user_sgpr_count", 5)? as u32;
    let least = 2 * (dispatch + queue + kernarg_ptr + dispatch_id) + private_size + res.user_sgprs;
    let most = target.definition().chip.user_sgprs;
    if user < least || user > most {
        return Err(Error::Asm(format!(
            ".amdhsa_user_sgpr_count {user}: not between {least} and {most}"
        )));
    }
    let rsrc1 = res.rsrc1
        | (block.bits("float_round_mode_32", 2)? as u32) << 12
        | (block.bits("float_round_mode_16_64", 2)? as u32) << 14
        | (block.bits("float_denorm_mode_32", 2)? as u32) << 16
        | (block.bits("float_denorm_mode_16_64", 2)? as u32) << 18
        | flag("fp16_overflow")? << 26;
    let rsrc2 = flag("enable_private_segment")?
        | user << 1
        | flag("system_sgpr_workgroup_id_x")? << 7
        | flag("system_sgpr_workgroup_id_y")? << 8
        | flag("system_sgpr_workgroup_id_z")? << 9
        | flag("system_sgpr_workgroup_info")? << 10
        | (block.bits("system_vgpr_workitem_id", 2)? as u32) << 11;
    let props = res.props
        | dispatch << 1
        | queue << 2
        | kernarg_ptr << 3
        | dispatch_id << 4
        | private_size << 6
        | flag("uses_dynamic_stack")? << 11;
    let mut d = [0u8; 64];
    let mut put = |at: usize, value: u32| d[at..at + 4].copy_from_slice(&value.to_le_bytes());
    put(0, block.bits("group_segment_fixed_size", 32)? as u32);
    put(4, block.bits("private_segment_fixed_size", 32)? as u32);
    put(8, block.bits("kernarg_size", 32)? as u32);
    put(44, res.rsrc3);
    put(48, rsrc1);
    put(52, rsrc2);
    put(56, (props & 0xFFFF) | (res.preload & 0xFFFF) << 16);
    Ok(d)
}
