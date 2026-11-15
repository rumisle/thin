//! Coverage-time check of every Interface descriptor field through thin's writer.

use std::collections::BTreeMap;

use thin_lift::interface::Interface;
use thin_target::{SFile, Target, COMMON};

struct Field {
    name: &'static str,
    at: usize,
    shift: u32,
    mask: u32,
}

const FIELDS: [Field; 17] = [
    Field {
        name: "group_segment_fixed_size",
        at: 0,
        shift: 0,
        mask: u32::MAX,
    },
    Field {
        name: "private_segment_fixed_size",
        at: 4,
        shift: 0,
        mask: u32::MAX,
    },
    Field {
        name: "kernarg_size",
        at: 8,
        shift: 0,
        mask: u32::MAX,
    },
    Field {
        name: "enable_private_segment",
        at: 52,
        shift: 0,
        mask: 1,
    },
    Field {
        name: "user_sgpr_count",
        at: 52,
        shift: 1,
        mask: 31,
    },
    Field {
        name: "system_sgpr_workgroup_id_x",
        at: 52,
        shift: 7,
        mask: 1,
    },
    Field {
        name: "system_sgpr_workgroup_id_y",
        at: 52,
        shift: 8,
        mask: 1,
    },
    Field {
        name: "system_sgpr_workgroup_id_z",
        at: 52,
        shift: 9,
        mask: 1,
    },
    Field {
        name: "system_sgpr_workgroup_info",
        at: 52,
        shift: 10,
        mask: 1,
    },
    Field {
        name: "system_vgpr_workitem_id",
        at: 52,
        shift: 11,
        mask: 3,
    },
    Field {
        name: "user_sgpr_dispatch_ptr",
        at: 56,
        shift: 1,
        mask: 1,
    },
    Field {
        name: "user_sgpr_queue_ptr",
        at: 56,
        shift: 2,
        mask: 1,
    },
    Field {
        name: "user_sgpr_kernarg_segment_ptr",
        at: 56,
        shift: 3,
        mask: 1,
    },
    Field {
        name: "user_sgpr_dispatch_id",
        at: 56,
        shift: 4,
        mask: 1,
    },
    Field {
        name: "user_sgpr_private_segment_size",
        at: 56,
        shift: 6,
        mask: 1,
    },
    Field {
        name: "wavefront_size32",
        at: 56,
        shift: 10,
        mask: 1,
    },
    Field {
        name: "uses_dynamic_stack",
        at: 56,
        shift: 11,
        mask: 1,
    },
];

impl Field {
    fn get(&self, descriptor: &[u8; 64]) -> u32 {
        let word = u32::from_le_bytes(
            descriptor[self.at..self.at + 4]
                .try_into()
                .expect("descriptor dword"),
        );
        (word >> self.shift) & self.mask
    }
}

#[derive(Default)]
pub struct Law {
    checked: u64,
    exact: u64,
    differences: BTreeMap<&'static str, u64>,
    examples: BTreeMap<&'static str, String>,
    errors: BTreeMap<String, u64>,
}

impl Law {
    /// Encodes from independent zero defaults, never from the input's saved
    /// residue. Only fields represented by Interface are compared.
    pub fn check(
        &mut self,
        interface: &Interface,
        original: &[u8; 64],
        target: &Target,
        context: &str,
    ) {
        self.checked += 1;
        let mut file = SFile::default();
        file.once
            .insert(".amdhsa_kernel".into(), ("interface_law".into(), 0));
        for key in COMMON.iter().chain(target.hooks().directives()) {
            file.block.insert((*key).into(), ("0".into(), 0));
        }
        interface.apply_to(&mut file);
        let encoded = match thin_target::interface_descriptor(&file, target) {
            Ok(d) => d,
            Err(e) => {
                *self.errors.entry(e.to_string()).or_default() += 1;
                return;
            }
        };
        let mut exact = true;
        for field in &FIELDS {
            let (before, after) = (field.get(original), field.get(&encoded));
            if before != after {
                exact = false;
                *self.differences.entry(field.name).or_default() += 1;
                self.examples.entry(field.name).or_insert_with(|| {
                    format!("{context}: original {before:#x}, written {after:#x}")
                });
            }
        }
        self.exact += u64::from(exact);
    }

    pub fn report(&self) {
        println!(
            "interface law: checked {}; exact {}; fields per interface {}",
            self.checked,
            self.exact,
            FIELDS.len()
        );
        for field in &FIELDS {
            println!(
                "interface field {}: differences {}",
                field.name,
                self.differences.get(field.name).copied().unwrap_or(0)
            );
            if let Some(example) = self.examples.get(field.name) {
                println!("  example: {example}");
            }
        }
        for (error, count) in &self.errors {
            println!("interface writer refusal {count}: {error}");
        }
    }

    pub fn passes(&self) -> bool {
        self.checked == self.exact
    }
}
