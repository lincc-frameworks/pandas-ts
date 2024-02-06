import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal

from pandas_ts import TsDtype
from pandas_ts.ts_ext_array import TsExtensionArray


def test_ts_ext_array_dtype():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = TsExtensionArray(struct_array)
    assert ext_array.dtype == TsDtype(struct_array.type)


def test_series_dtype():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = TsExtensionArray(struct_array)
    series = pd.Series(ext_array)
    assert series.dtype == TsDtype(struct_array.type)


def test_series_built_with_dtype():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    dtype = TsDtype(struct_array.type)
    series = pd.Series(struct_array, dtype=dtype)
    assert isinstance(series.array, TsExtensionArray)


def test_series_built_from_dict():
    data = [
        {"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]},
        {"a": [1, 2, 1], "b": [-3.0, -4.0, -5.0]},
    ]
    dtype = TsDtype.from_fields({"a": pa.uint8(), "b": pa.float64()})
    series = pd.Series(data, dtype=dtype)

    assert isinstance(series.array, TsExtensionArray)
    assert series.array.dtype == dtype

    desired_ext_array = TsExtensionArray(
        pa.StructArray.from_arrays(
            arrays=[
                pa.array([np.array([1, 2, 3]), np.array([1, 2, 1])], type=pa.list_(pa.uint8())),
                pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
            ],
            names=["a", "b"],
        )
    )
    assert_series_equal(series, pd.Series(desired_ext_array))


# Test exception raises for wrong dtype
@pytest.mark.parametrize(
    "data",
    [
        # Must be struct
        [
            1,
            2,
            3,
        ],
        # Must be struct
        {"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]},
        # Lists of the same object must have the same length for each field
        [{"a": [1, 2, 3], "b": [-4.0, -5.0, -6.0]}, {"a": [1, 2, 1], "b": [-3.0, -4.0]}],
        # Struct fields must be lists
        [{"a": 1, "b": [-4.0, -5.0, -6.0]}, {"a": 2, "b": [-3.0, -4.0, -5.0]}],
    ],
)
def test_series_built_raises(data):
    pa_array = pa.array(data)
    with pytest.raises(ValueError):
        _array = TsExtensionArray(pa_array)


def test_list_offsets():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([1, 2, 1])], type=pa.list_(pa.uint8())),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = TsExtensionArray(struct_array)

    desired = pa.chunked_array([pa.array([0, 3, 6])])
    assert_array_equal(ext_array.list_offsets, desired)


def test___getitem__():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[100, 101])

    second_row_as_df = series[101]
    assert_frame_equal(
        second_row_as_df, pd.DataFrame({"a": np.array([1.0, 2.0, 1.0]), "b": -np.array([3.0, 4.0, 5.0])})
    )


def test_series_apply_udf_argument():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[100, 101])

    series_of_dfs = series.apply(lambda x: x)
    assert_frame_equal(
        series_of_dfs.iloc[0], pd.DataFrame({"a": np.array([1.0, 2.0, 3.0]), "b": -np.array([4.0, 5.0, 6.0])})
    )


def test___iter__():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[100, 101])

    # Check last df only
    df = list(series)[-1]
    assert_frame_equal(df, pd.DataFrame({"a": np.array([1.0, 2.0, 1.0]), "b": -np.array([3.0, 4.0, 5.0])}))


def test_field_names():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = TsExtensionArray(struct_array)

    assert ext_array.field_names == ["a", "b"]


def test_flat_length():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0, 2.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = TsExtensionArray(struct_array)

    assert ext_array.flat_length == 7


def test_set_flat_field_new_field_scalar():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0, 2.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = TsExtensionArray(struct_array)

    ext_array.set_flat_field("c", "YES")

    desired_struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0, 2.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
            pa.array([["YES"] * 3, ["YES"] * 4]),
        ],
        names=["a", "b", "c"],
    )
    desired = TsExtensionArray(desired_struct_array)

    assert_series_equal(pd.Series(ext_array), pd.Series(desired))


def test_set_flat_field_replace_field_array():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0, 4.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = TsExtensionArray(struct_array)

    ext_array.set_flat_field("b", [True, False, True, False, True, False, True])

    desired_struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0, 4.0])]),
            pa.array([np.array([True, False, True]), np.array([False, True, False, True])]),
        ],
        names=["a", "b"],
    )
    desired = TsExtensionArray(desired_struct_array)

    assert_series_equal(pd.Series(ext_array), pd.Series(desired))


def test_set_list_field_new_field():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0, 2.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = TsExtensionArray(struct_array)

    ext_array.set_list_field("c", [["x", "y", "z"], ["x1", "x2", "x3", "x4"]])

    desired_struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0, 2.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
            pa.array([np.array(["x", "y", "z"]), np.array(["x1", "x2", "x3", "x4"])]),
        ],
        names=["a", "b", "c"],
    )
    desired = TsExtensionArray(desired_struct_array)

    assert_series_equal(pd.Series(ext_array), pd.Series(desired))


def test_set_list_field_replace_field():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0, 2.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
        ],
        names=["a", "b"],
    )
    ext_array = TsExtensionArray(struct_array)

    ext_array.set_list_field("b", [["x", "y", "z"], ["x1", "x2", "x3", "x4"]])

    desired_struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0, 2.0])]),
            pa.array([np.array(["x", "y", "z"]), np.array(["x1", "x2", "x3", "x4"])]),
        ],
        names=["a", "b"],
    )
    desired = TsExtensionArray(desired_struct_array)

    assert_series_equal(pd.Series(ext_array), pd.Series(desired))


def test_delete_field():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0, 2.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
            pa.array([np.array(["x", "y", "z"]), np.array(["x1", "x2", "x3", "x4"])]),
        ],
        names=["a", "b", "c"],
    )
    ext_array = TsExtensionArray(struct_array)

    ext_array.delete_field("c")

    desired_struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0, 2.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
        ],
        names=["a", "b"],
    )
    desired = TsExtensionArray(desired_struct_array)

    assert_series_equal(pd.Series(ext_array), pd.Series(desired))


def test_delete_last_field_raises():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0, 2.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0, 6.0])]),
            pa.array([np.array(["x", "y", "z"]), np.array(["x1", "x2", "x3", "x4"])]),
        ],
        names=["a", "b", "c"],
    )
    ext_array = TsExtensionArray(struct_array)

    ext_array.delete_field("a")
    assert ext_array.field_names == ["b", "c"]

    ext_array.delete_field("c")
    assert ext_array.field_names == ["b"]

    with pytest.raises(ValueError):
        ext_array.delete_field("b")
