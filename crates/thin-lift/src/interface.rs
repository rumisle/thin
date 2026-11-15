//! The inverse of writing a kernel's descriptor and metadata interface.
//!
//! Only interface fields are consumed. Resource allocation, entry offsets,
//! reserved bits and unrecognized metadata stay in the residue. The law is
//! `residue.write(interface, template, target) == descriptor`: the writer
//! supplies the consumed fields, not a saved copy of them.

use thin_formats::Value;
use thin_target::{SFile, Target};

use crate::object::ObjectError;

/// The metadata's argument value kind, including extension spellings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArgumentKind {
    /// Bytes passed directly in the argument segment.
    ByValue,
    /// A pointer into global memory.
    GlobalBuffer,
    /// A pointer into dynamically allocated LDS.
    DynamicSharedPointer,
    /// An ABI-defined implicit argument or an extension kind.
    Other(String),
}

/// One argument in the kernel argument segment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Argument {
    /// Optional source name; an absent name is not an empty name.
    pub name: Option<String>,
    /// Byte offset in the argument segment.
    pub offset: u32,
    /// Size in bytes.
    pub size: u32,
    /// The metadata's value kind.
    pub kind: ArgumentKind,
}

/// A runtime value delivered in the user SGPRs, in ABI order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UserValue {
    /// Dispatch packet pointer (two dwords).
    DispatchPointer,
    /// Queue pointer (two dwords).
    QueuePointer,
    /// Kernel argument segment pointer (two dwords).
    KernargPointer,
    /// Dispatch identifier (two dwords).
    DispatchId,
    /// Private segment size (one dword).
    PrivateSegmentSize,
}

impl UserValue {
    fn directive(self) -> &'static str {
        match self {
            Self::DispatchPointer => "user_sgpr_dispatch_ptr",
            Self::QueuePointer => "user_sgpr_queue_ptr",
            Self::KernargPointer => "user_sgpr_kernarg_segment_ptr",
            Self::DispatchId => "user_sgpr_dispatch_id",
            Self::PrivateSegmentSize => "user_sgpr_private_segment_size",
        }
    }

    /// Width of this runtime value in SGPR dwords.
    pub fn dwords(self) -> u32 {
        if self == Self::PrivateSegmentSize {
            1
        } else {
            2
        }
    }
}

const USER_VALUES: [(UserValue, u32); 5] = [
    (UserValue::DispatchPointer, 1),
    (UserValue::QueuePointer, 2),
    (UserValue::KernargPointer, 3),
    (UserValue::DispatchId, 4),
    (UserValue::PrivateSegmentSize, 6),
];

/// An initialized range of user SGPRs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UserSgpr {
    /// The first SGPR, without alignment padding.
    pub first: u32,
    /// The runtime value; its kind fixes the width.
    pub value: UserValue,
}

/// The private segment properties from the descriptor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrivateSegment {
    /// Fixed allocation in bytes per workitem.
    pub size: u32,
    /// Private segment access is enabled.
    pub enabled: bool,
    /// The kernel can use a dynamic stack.
    pub dynamic_stack: bool,
}

/// The typed entry interface of one kernel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Interface {
    /// Arguments in metadata order, not sorted by offset.
    pub arguments: Vec<Argument>,
    /// Total kernel argument segment size in bytes.
    pub kernarg_size: u32,
    /// Size stored in descriptor bytes 8..12, independently of metadata.
    /// Older shipped objects leave this word zero despite nonempty arguments.
    pub descriptor_kernarg_size: u32,
    /// Initialized user SGPR ranges in ABI order.
    pub user_sgprs: Vec<UserSgpr>,
    /// Reserved user SGPR count, including any unnamed tail.
    pub user_sgpr_count: u32,
    /// Workgroup-id enables in x, y, z order, packed after user SGPRs.
    pub workgroup_ids: [bool; 3],
    /// Whether the workgroup information SGPR follows the ids.
    pub workgroup_info: bool,
    /// Raw descriptor workitem-id component mode (0, 1 or 2).
    pub workitem_id_components: u8,
    /// Fixed LDS allocation in bytes per workgroup.
    pub lds_size: u32,
    /// Number of lanes per wave (32 or 64).
    pub wave_size: u32,
    /// Private segment properties.
    pub private_segment: PrivateSegment,
}

// These masks name exactly the fields consumed by read_interface.
const RSRC2_MASK: u32 = 1 | (31 << 1) | (63 << 7);
const PROPS_MASK: u32 =
    (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 6) | (1 << 10) | (1 << 11);

/// The descriptor and note fields not represented by an [`Interface`].
#[derive(Clone, Debug, PartialEq)]
pub struct InterfaceResidue {
    /// Descriptor bytes with every consumed bit cleared.
    pub descriptor: [u8; 64],
    /// Kernel metadata with `.args` removed.
    pub metadata: Value,
    /// Unconsumed fields of each argument, in metadata order.
    pub argument_metadata: Vec<Value>,
}

fn word(d: &[u8; 64], at: usize) -> u32 {
    u32::from_le_bytes(d[at..at + 4].try_into().expect("four descriptor bytes"))
}

fn number(v: &Value, key: &str) -> Result<u32, ObjectError> {
    match v.get(key) {
        Some(Value::Int(n)) => {
            u32::try_from(*n).map_err(|_| ObjectError(format!("{key}: outside the u32 range")))
        }
        _ => Err(ObjectError(format!("{key}: expected an integer"))),
    }
}

fn without(v: &Value, keys: &[&str]) -> Value {
    match v {
        Value::Map(entries) => Value::Map(
            entries
                .iter()
                .filter(|(k, _)| !keys.contains(&k.as_str()))
                .cloned()
                .collect(),
        ),
        _ => v.clone(),
    }
}

/// Recovers the descriptor's interface and the unconsumed fields of the
/// per-kernel AMDGPU metadata map. Unknown argument kinds are preserved.
/// Missing or malformed argument descriptions are refused, not invented.
pub fn read_interface(
    descriptor: &[u8; 64],
    metadata: &Value,
) -> Result<(Interface, InterfaceResidue), ObjectError> {
    let mut arguments = Vec::new();
    let mut argument_metadata = Vec::new();
    let descriptor_kernarg_size = word(descriptor, 8);
    let kernarg_size = match metadata.get(".kernarg_segment_size") {
        Some(_) => number(metadata, ".kernarg_segment_size")?,
        None => descriptor_kernarg_size,
    };
    let entries = match metadata.get(".args") {
        Some(Value::Array(a)) => a.as_slice(),
        None if matches!(metadata, Value::Map(_)) => &[],
        _ => return Err(ObjectError(".args: expected an array".into())),
    };
    for arg in entries {
        let name = match arg.get(".name") {
            Some(Value::Str(s)) => Some(s.clone()),
            None => None,
            _ => return Err(ObjectError(".name: expected a string".into())),
        };
        let offset = number(arg, ".offset")?;
        let size = number(arg, ".size")?;
        if offset
            .checked_add(size)
            .is_none_or(|end| end > kernarg_size)
        {
            return Err(ObjectError(
                "argument extends past the kernarg segment".into(),
            ));
        }
        let kind = match arg.get(".value_kind") {
            Some(Value::Str(s)) => match s.as_str() {
                "by_value" => ArgumentKind::ByValue,
                "global_buffer" => ArgumentKind::GlobalBuffer,
                "dynamic_shared_pointer" => ArgumentKind::DynamicSharedPointer,
                _ => ArgumentKind::Other(s.clone()),
            },
            _ => return Err(ObjectError(".value_kind: expected a string".into())),
        };
        arguments.push(Argument {
            name,
            offset,
            size,
            kind,
        });
        argument_metadata.push(without(arg, &[".name", ".offset", ".size", ".value_kind"]));
    }
    let rsrc2 = word(descriptor, 52);
    let props = word(descriptor, 56);
    let mut first = 0;
    let mut user_sgprs = Vec::new();
    for (value, bit) in USER_VALUES {
        if props & (1 << bit) != 0 {
            user_sgprs.push(UserSgpr { first, value });
            first += value.dwords();
        }
    }
    let user_sgpr_count = (rsrc2 >> 1) & 31;
    if first > user_sgpr_count {
        return Err(ObjectError(
            "user SGPR values exceed the reserved count".into(),
        ));
    }
    let workitem_id_components = ((rsrc2 >> 11) & 3) as u8;
    if workitem_id_components == 3 {
        return Err(ObjectError("reserved workitem-id component mode 3".into()));
    }
    let interface = Interface {
        arguments,
        kernarg_size,
        descriptor_kernarg_size,
        user_sgprs,
        user_sgpr_count,
        workgroup_ids: std::array::from_fn(|axis| rsrc2 & (1 << (7 + axis)) != 0),
        workgroup_info: rsrc2 & (1 << 10) != 0,
        workitem_id_components,
        lds_size: word(descriptor, 0),
        wave_size: if props & (1 << 10) != 0 { 32 } else { 64 },
        private_segment: PrivateSegment {
            size: word(descriptor, 4),
            enabled: rsrc2 & 1 != 0,
            dynamic_stack: props & (1 << 11) != 0,
        },
    };
    let mut remaining = *descriptor;
    remaining[..12].fill(0);
    remaining[52..56].copy_from_slice(&(rsrc2 & !RSRC2_MASK).to_le_bytes());
    remaining[56..60].copy_from_slice(&(props & !PROPS_MASK).to_le_bytes());
    Ok((
        interface,
        InterfaceResidue {
            descriptor: remaining,
            metadata: without(metadata, &[".args"]),
            argument_metadata,
        },
    ))
}

impl Interface {
    /// Supplies the interface directives of a writer input. Resource
    /// allocation and other target directives remain the caller's choice.
    /// The target writer still validates all directives (including wave size).
    pub fn apply_to(&self, file: &mut SFile) {
        let mut put = |key: &str, value: u32| {
            file.block.insert(key.to_owned(), (value.to_string(), 0));
        };
        put("group_segment_fixed_size", self.lds_size);
        put("private_segment_fixed_size", self.private_segment.size);
        put("kernarg_size", self.descriptor_kernarg_size);
        put(
            "enable_private_segment",
            self.private_segment.enabled.into(),
        );
        put(
            "uses_dynamic_stack",
            self.private_segment.dynamic_stack.into(),
        );
        put("wavefront_size32", u32::from(self.wave_size == 32));
        put("user_sgpr_count", self.user_sgpr_count);
        for (value, _) in USER_VALUES {
            put(
                value.directive(),
                u32::from(self.user_sgprs.iter().any(|s| s.value == value)),
            );
        }
        for (axis, enabled) in ["x", "y", "z"].into_iter().zip(self.workgroup_ids) {
            put(&format!("system_sgpr_workgroup_id_{axis}"), enabled.into());
        }
        put("system_sgpr_workgroup_info", self.workgroup_info.into());
        put(
            "system_vgpr_workitem_id",
            self.workitem_id_components.into(),
        );
    }
}

impl InterfaceResidue {
    /// Runs thin's descriptor writer with the recovered interface, then
    /// restores unconsumed bits. No consumed bit is saved in this residue.
    pub fn write(
        &self,
        interface: &Interface,
        template: &SFile,
        target: &Target,
    ) -> Result<[u8; 64], thin_target::Error> {
        let mut file = template.clone();
        interface.apply_to(&mut file);
        let encoded = thin_target::descriptor(&file, target)?;
        let mut out = self.descriptor;
        out[..12].copy_from_slice(&encoded[..12]);
        for (at, mask) in [(52, RSRC2_MASK), (56, PROPS_MASK)] {
            let bits = word(&out, at) | (word(&encoded, at) & mask);
            out[at..at + 4].copy_from_slice(&bits.to_le_bytes());
        }
        Ok(out)
    }
}
