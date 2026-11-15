"""Verifier programs and native form descriptions.

Structural constructors supply input programs; thin supplies errors and form
facts. Descriptions identify the verifier rule exercised by each case.
"""

from __future__ import annotations

from thin import ir
from thin.target import Flags, target
from tools.golden.ir_inputs import _base_program, _instructions

VERIFY_BEHAVIORS = {
    "valid_branch_loop": "a valid arithmetic, branch, label and end program",
    "missing_add_operand": "a fixed-arity instruction with too few operands",
    "scalar_destination": "a scalar value in a vector destination",
    "wide_add_destination": "a two-dword value in a fixed one-dword destination",
    "load_address_widths": "both accepted widths of a load address",
    "register_range_overflow": "a register range extending beyond its value",
    "named_vector_destination": "a named non-vector register in a vector destination",
    "off_address": "the physical off register accepted by a width-choice address slot",
    "immediate_destination": "an immediate in a vector destination",
    "immediate_source": "an immediate accepted in a vector source",
    "block_source": "a block reference in a vector source",
    "unsupported_modifier": "nonzero modifiers on a slot that does not allow them",
    "zero_modifiers": "zero modifier bytes allowed on an unmodified slot",
    "unresolved_branch": "a branch reference absent from the program",
    "forward_branch": "a forward branch resolved by a later label",
    "unknown_add_field": "an unknown field on a form with no fields",
    "unknown_ds_field": "an unknown field with the form's available fields listed",
    "unsigned_offset_overflow": "an unsigned integer one above its maximum",
    "negative_unsigned_offset": "a negative integer in an unsigned field",
    "boolean_integer_offset": "a bool refused by an integer field",
    "signed_offset_underflow": "a signed integer one below its minimum",
    "float_integer_offset": "a non-integer object numerically inside an integer range",
    "integer_flag": "a non-boolean value in a flag field",
    "oversized_bits": "a bit tuple wider than its field",
    "nonbinary_bits": "a bit tuple containing a non-binary member",
    "empty_bits": "an empty bit tuple",
    "invalid_wait_counter": "a choice value outside its declared alternatives",
    "wait_counter_text": "a valid wait choice and optional text",
    "missing_wait_count": "one required pseudo field absent",
    "wait_count_overflow": "an eight-bit wait count one above its maximum",
    "missing_wait_counter": "the required wait counter absent",
    "integer_label_name": "an integer in a text field",
    "text_region_tuple": "text in a tuple field",
    "named_regions": "a names tuple accepted by a variadic wait marker",
    "empty_variadic_wait": "an empty operand list on a variadic pseudo instruction",
    "mixed_wait_registers": "value and physical register operands on a variadic pseudo instruction",
    "immediate_wait_operand": "an immediate in an any-value pseudo slot",
    "extra_assume_operand": "a fixed pseudo instruction with an extra operand",
    "assume_bound_overflow": "a 32-bit assumption bound one above its maximum",
    "label_with_operand": "a non-variadic label with an operand",
    "nonbarrier_fence": "a fence attribute on a non-barrier instruction",
    "nonmemory_places": "a places attribute on a non-memory instruction",
    "empty_places": "an empty places attribute accepted as absent",
    "nonbranch_waves": "a wave-taken attribute on a non-branch instruction",
    "nonmemory_lds": "an LDS-access attribute on a non-memory instruction",
    "nonatomic_published": "a published attribute on a non-atomic instruction",
    "printable_wait_marker": "an otherwise valid wait marker left for the printer",
    "first_instruction_error": (
        "the earliest instruction's branch error precedes a later operand error"
    ),
    "operand_error_precedence": "an operand error precedes an error in an unknown field",
}


def _form_record(form):
    """A form declaration read from the native target, with no expected text."""
    flags = Flags.names(form.flags)
    return {
        "slots": [
            {
                "kind": slot.kind,
                "name": slot.name,
                "dwords": slot.dwords,
                "widths": list(slot.widths),
                "modifiers": slot.modifiers,
            }
            for slot in form.slots
        ],
        "fields": [
            {
                "name": field.name,
                "kind": field.kind,
                "bits": field.bits,
                "signed": field.signed,
                "choices": list(field.choices),
            }
            for field in form.fields
        ],
        "flags": [] if flags == "NONE" else flags.split("|"),
    }


def verify_data():
    """Build every program and ask the native verifier for its first refusal."""
    t = target("gfx1201")
    cases = []
    forms = {}
    for name, behavior in VERIFY_BEHAVIORS.items():
        p, instructions = _instructions(t, name)
        p.body.extend(instructions)
        for i in instructions:
            if i.desc.id >= 0:
                forms[f"{i.desc.name}/{len(i.desc.slots)}"] = _form_record(i.desc)
        printable = name == "printable_wait_marker"
        try:
            ir.verify(p, printable)
            error = None
        except ir.Diag as raised:
            error = raised.msg
        cases.append(
            {
                "name": name,
                "program": behavior,
                "printable": printable,
                "error": error,
            }
        )
    return {
        "target": t.name,
        "values": {
            v.name: f"{v.file}, {v.n} dword" + ("s" if v.n != 1 else "")
            for v in _base_program(t).values
        },
        "note": (
            "Each case constructs a fresh IR Program. error is the native verifier's Diag.msg; "
            "null means verification succeeded. forms are read from the native target forms "
            "actually used by the authored programs. Values retain stable ids 0 through 4."
        ),
        "forms": dict(sorted(forms.items())),
        "cases": cases,
    }
