"""Each target's XML member, fix tables and operand-table rules."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TargetRules:
    """What a target supplies to the generator."""

    member: str
    """The target's XML member of the bundle."""
    arch: str
    """The architecture key of the fix tables."""
    vector: tuple[str, ...]
    """Encoding-name prefixes of vector instructions."""
    flat: tuple[str, ...]
    """Prefixes of the flat family."""
    rtn_encodings: tuple[str, ...]
    """Prefixes whose atomics key a returning form with ``+rtn``."""
    tied: str
    """A regular expression on the mnemonic: operand 0 is read and written."""
    no_exec: tuple[str, ...]
    """Mnemonics that ignore EXEC."""
    mask_dwords: int
    """1 for wave32, 2 for wave64."""
    unwritten: dict[str, str]
    """Operand type to register name: explicit in the XML, absent from the text."""
    skip: tuple[str, ...]
    """Encoding prefixes left out."""
    hand: tuple[tuple[str, int], ...] = field(default=())
    """Rows given by hand, by (mnemonic, count)."""
    alias: dict[str, str] = field(default_factory=dict)
    """Other spellings."""
    lds_dma: bool = False
    """Whether a buffer load of 4 operands also has a DMA-into-LDS form."""


TARGETS = {
    "gfx1201": TargetRules(
        member="amdgpu_isa_rdna4.xml",
        arch="rdna4",
        vector=(
            "ENC_VOP",
            "VOP",
            "ENC_VDS",
            "ENC_VFLAT",
            "ENC_VGLOBAL",
            "ENC_VSCRATCH",
            "ENC_VBUFFER",
        ),
        flat=("ENC_VFLAT", "ENC_VGLOBAL", "ENC_VSCRATCH"),
        rtn_encodings=("ENC_VBUFFER",),
        tied=r"^(v_fmac_|v_dot\d+c_|v_pk_fmac_|v_swmmac_|v_cvt(_sr)?(_pk)?_(fp8|bf8)_|v_writelane_|s_bitset[01]_)",
        no_exec=("v_readlane_b32", "v_writelane_b32"),
        mask_dwords=1,
        unwritten={"OPR_EXEC": "exec"},
        skip=("ENC_VEXPORT", "ENC_VIMAGE", "ENC_VSAMPLE", "ENC_VINTERP", "ENC_VDSDIR"),
        hand=(
            ("s_wait_loadcnt", 0),
            ("s_wait_storecnt", 0),
            ("s_wait_dscnt", 0),
            ("s_wait_kmcnt", 0),
            ("s_wait_alu", 0),
        ),
    ),
}
"""The targets the generator knows."""
