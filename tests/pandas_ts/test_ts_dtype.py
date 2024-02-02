import pyarrow as pa
import pytest

from pandas_ts import TsDtype
from pandas_ts.ts_ext_array import TsExtensionArray


@pytest.mark.parametrize(
    "pyarrow_dtype",
    [
        pa.struct([pa.field("a", pa.list_(pa.int64()))]),
        pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))]),
        pa.struct(
            [
                pa.field("a", pa.list_(pa.int64())),
                pa.field("b", pa.list_(pa.struct([pa.field("c", pa.int64())]))),
            ]
        ),
    ],
)
def test_from_pyarrow_dtype(pyarrow_dtype):
    dtype = TsDtype(pyarrow_dtype)
    assert dtype.pyarrow_dtype == pyarrow_dtype


@pytest.mark.parametrize(
    "pyarrow_dtype",
    [
        pa.int64(),
        pa.list_(pa.int64()),
        pa.list_(pa.struct([pa.field("a", pa.int64())])),
        pa.struct([pa.field("a", pa.int64())]),
        pa.struct([pa.field("a", pa.int64()), pa.field("b", pa.float64())]),
        pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.float64())]),
    ],
)
def test_from_pyarrow_dtype_raises(pyarrow_dtype):
    with pytest.raises(ValueError):
        TsDtype(pyarrow_dtype)


def test_from_fields():
    fields = {"a": pa.int64(), "b": pa.float64()}
    dtype = TsDtype.from_fields(fields)
    assert dtype.pyarrow_dtype == pa.struct(
        [pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.float64()))]
    )


@pytest.mark.parametrize(
    "fields",
    [
        {"a": pa.int64(), "b": pa.float64()},
        {"a": pa.int64(), "b": pa.float64(), "c": pa.int64()},
        {"a": pa.string(), "b": pa.float64()},
        # Nested / parametric types are not implemented.
        # {"a": pa.list_(pa.int64()), "b": pa.float64()},
        # {"a": pa.list_(pa.int64()), "b": pa.list_(pa.string())},
        # {"a": pa.struct([pa.field("a", pa.int64())]), "b": pa.list_(pa.int64())},
    ],
)
def test_name_vs_construct_from_string(fields):
    dtype = TsDtype.from_fields(fields)
    assert dtype == TsDtype.construct_from_string(dtype.name)


def test_construct_array_type():
    assert TsDtype.construct_array_type() is TsExtensionArray
