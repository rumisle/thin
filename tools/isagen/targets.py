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
    renames: dict[str, str] = field(default_factory=dict)
    """XML instruction names replaced by their assembler spellings."""
    drop: tuple[tuple[str, str], ...] = field(default=())
    """(instruction, encoding) pairs present in the XML that the target lacks."""
    lds_dma: tuple[str, ...] = field(default=())
    """Buffer-load mnemonics that also have a 3-operand DMA-into-LDS form."""
    saddr_default: int = 124
    """Default 7-bit saddr field value (124 = null on RDNA4, 127 = off on CDNA3)."""
    soffset_default: int = 124
    """Default 7-bit soffset field value (124 = null on RDNA4, 0 on CDNA3)."""


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
    "gfx942": TargetRules(
        member="amdgpu_isa_cdna3.xml",
        arch="cdna3",
        vector=(
            "ENC_VOP",
            "VOP",
            "ENC_DS",
            "ENC_FLAT",
            "ENC_MUBUF",
            "ENC_MTBUF",
        ),
        flat=("ENC_FLAT",),
        rtn_encodings=("ENC_MUBUF", "ENC_MTBUF"),
        tied=r"^(v_fmac_|v_dot\d+c_|v_pk_fmac_|v_smfmac_|v_cvt_pkaccum_|v_writelane_|s_bitset[01]_)",
        no_exec=("v_readlane_b32", "v_writelane_b32"),
        mask_dwords=2,
        unwritten={},
        skip=("ENC_MTBUF",),
        hand=(("s_waitcnt", 0),),
        renames={
            "v_accvgpr_read": "v_accvgpr_read_b32",
            "v_accvgpr_write": "v_accvgpr_write_b32",
            "v_mad_mix_f32": "v_fma_mix_f32",
            "v_mad_mixlo_f16": "v_fma_mixlo_f16",
            "v_mad_mixhi_f16": "v_fma_mixhi_f16",
        },
        drop=(
            ("s_set_valu_coexec_mode", "ENC_SOPP"),
            ("v_readfirstlane_b32", "ENC_VOP3"),
            ("v_swap_b32", "ENC_VOP3"),
            ("v_accvgpr_mov_b32", "ENC_VOP3"),
        ),
        lds_dma=(
            "buffer_load_dword",
            "buffer_load_format_x",
            "buffer_load_sbyte",
            "buffer_load_sshort",
            "buffer_load_ubyte",
            "buffer_load_ushort",
        ),
        saddr_default=127,
        soffset_default=0,
    ),
    "gfx950": TargetRules(
        member="amdgpu_isa_cdna4.xml",
        arch="cdna4",
        vector=(
            "ENC_VOP",
            "VOP",
            "ENC_DS",
            "ENC_FLAT",
            "ENC_MUBUF",
            "ENC_MTBUF",
        ),
        flat=("ENC_FLAT",),
        rtn_encodings=("ENC_MUBUF", "ENC_MTBUF"),
        tied=(
            r"^(v_fmac_|v_mac_|v_dot\d+c_|v_pk_fmac_|v_smfmac_|v_cvt_pkaccum_|"
            r"v_cvt_pk_(fp8|bf8)_f32|v_cvt_sr_(fp8|bf8|f16|bf16)_f32|"
            r"v_cvt_scalef32_(f16_(fp8|bf8)|pk_(fp8|bf8|fp4)_|sr_(fp8|bf8)_|sr_pk_fp4_)|"
            r"v_writelane_|s_bitset[01]_|s_(addk|mulk|cmovk)_i32|s_cmov_b|s_(buffer_)?atomic_)"
        ),
        no_exec=("v_readlane_b32", "v_writelane_b32"),
        mask_dwords=2,
        unwritten={},
        skip=("ENC_MTBUF",),
        hand=(("s_waitcnt", 0),),
        renames={
            "v_accvgpr_read": "v_accvgpr_read_b32",
            "v_accvgpr_write": "v_accvgpr_write_b32",
            "v_mad_mix_f32": "v_fma_mix_f32",
            "v_mad_mixlo_f16": "v_fma_mixlo_f16",
            "v_mad_mixhi_f16": "v_fma_mixhi_f16",
        },
        drop=(
            ("s_set_valu_coexec_mode", "ENC_SOPP"),
            ("v_readfirstlane_b32", "ENC_VOP3"),
            ("v_swap_b32", "ENC_VOP3"),
            ("v_accvgpr_mov_b32", "ENC_VOP3"),
        ),
        lds_dma=(
            "buffer_load_dword",
            "buffer_load_format_x",
            "buffer_load_sbyte",
            "buffer_load_sshort",
            "buffer_load_ubyte",
            "buffer_load_ushort",
        ),
        saddr_default=127,
        soffset_default=0,
    ),
}
"""The targets the generator knows."""
