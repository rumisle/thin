"""The experimental checker's layout and record bridge."""

from thin.emu import gemm


def test_records_round_trip_scalar_and_tuple_columns():
    with gemm.fresh_tables():
        for n in (3, (3, 4)):
            tag = gemm.record_tag(2, n, {0, 1}, pid=7)
            assert gemm.record_of(tag) == (2, n, frozenset({0, 1}), False)
            assert gemm.record_pid(tag) == 7
        assert gemm.poison_tag("invalid cell") < 0
        a = gemm.record_tag(2, 3, {0})
        b = gemm.record_tag(2, 3, {1})
        assert gemm.record_of(gemm.union_records([a], [b])[0]) == (2, 3, frozenset({0, 1}), False)


def test_rowmajor_decode_byte_coordinates():
    layout = gemm.rowmajor("A", 16, elem_bytes=2)
    decoded = layout.decode([0, 1, 2, 16, 17])
    assert decoded["row"].tolist() == [0, 0, 0, 1, 1]
    assert decoded["k"].tolist() == [0, 0, 1, 0, 0]
    assert decoded["sub"].tolist() == [0, 1, 0, 0, 1]
