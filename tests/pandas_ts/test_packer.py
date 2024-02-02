import numpy as np
import pandas as pd
import pyarrow as pa
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal

from pandas_ts import packer


def test_pack_df():
    df = pd.DataFrame(
        data={
            "a": [7, 8, 9, 1, 2, 3, 4, 5, 6],
            "b": [0, 1, 0, 0, 1, 0, 1, 0, 1],
        },
        index=[4, 4, 4, 1, 1, 2, 2, 3, 3],
    )
    actual = packer.pack_df(df, name="struct")

    desired = pd.DataFrame(
        data={
            "a": pd.Series(
                data=[
                    np.array([1, 2]),
                    np.array([3, 4]),
                    np.array([5, 6]),
                    np.array([7, 8, 9]),
                ],
                dtype=pd.ArrowDtype(pa.list_(pa.int64())),
                index=[1, 2, 3, 4],
            ),
            "b": pd.Series(
                data=[
                    np.array([0, 1]),
                    np.array([0, 1]),
                    np.array([0, 1]),
                    np.array([0, 1, 0]),
                ],
                dtype=pd.ArrowDtype(pa.list_(pa.int64())),
                index=[1, 2, 3, 4],
            ),
            "struct": pd.Series(
                data=[
                    (np.array([1, 2]), np.array([0, 1])),
                    (np.array([3, 4]), np.array([0, 1])),
                    (np.array([5, 6]), np.array([0, 1])),
                    (np.array([7, 8, 9]), np.array([0, 1, 0])),
                ],
                dtype=pd.ArrowDtype(
                    pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.int64()))])
                ),
                index=[1, 2, 3, 4],
            ),
        },
    )

    assert_frame_equal(actual, desired)


def test_pack_df_into_structs():
    df = pd.DataFrame(
        data={
            "a": [7, 8, 9, 1, 2, 3, 4, 5, 6],
            "b": [0, 1, 0, 0, 1, 0, 1, 0, 1],
        },
        index=[4, 4, 4, 1, 1, 2, 2, 3, 3],
    )
    actual = packer.pack_df_into_structs(df)

    desired = pd.Series(
        data=[
            (np.array([1, 2]), np.array([0, 1])),
            (np.array([3, 4]), np.array([0, 1])),
            (np.array([5, 6]), np.array([0, 1])),
            (np.array([7, 8, 9]), np.array([0, 1, 0])),
        ],
        index=[1, 2, 3, 4],
        dtype=pd.ArrowDtype(
            pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.int64()))])
        ),
    )

    assert_series_equal(actual, desired)


def test_pack_sorted_df_into_struct():
    df = pd.DataFrame(
        data={
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "b": [0, 1, 0, 1, 0, 1, 0, 1, 0],
        },
        index=[1, 1, 2, 2, 3, 3, 4, 4, 4],
    )
    actual = packer.pack_sorted_df_into_struct(df)

    desired = pd.Series(
        data=[
            (np.array([1, 2]), np.array([0, 1])),
            (np.array([3, 4]), np.array([0, 1])),
            (np.array([5, 6]), np.array([0, 1])),
            (np.array([7, 8, 9]), np.array([0, 1, 0])),
        ],
        index=[1, 2, 3, 4],
        dtype=pd.ArrowDtype(
            pa.struct([pa.field("a", pa.list_(pa.int64())), pa.field("b", pa.list_(pa.int64()))])
        ),
    )

    assert_series_equal(actual, desired)


def test_view_packed_df_as_struct_series():
    packed_df = pd.DataFrame(
        data={
            "a": [
                np.array([1, 2]),
                np.array([3, 4]),
                np.array([5, 6]),
                np.array([7, 8, 9]),
            ],
            "b": [
                np.array([0, 1]),
                np.array([0, 1]),
                np.array([0, 1]),
                np.array([0, 1, 0]),
            ],
        },
        index=[1, 2, 3, 4],
        dtype=pd.ArrowDtype(pa.list_(pa.int64())),
    )
    series = packer.view_packed_df_as_struct_series(packed_df)

    for field_name in packed_df.columns:
        assert_series_equal(series.struct.field(field_name), packed_df[field_name])


def test_view_sorted_df_as_nested_arrays():
    flat_df = pd.DataFrame(
        data={
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "b": [0, 1, 0, 1, 0, 1, 0, 1, 0],
        },
        index=[1, 1, 2, 2, 3, 3, 4, 4, 4],
    )
    nested_df = packer.view_sorted_df_as_nested_arrays(flat_df)

    assert_array_equal(nested_df.index, [1, 2, 3, 4])

    desired_nested = pd.DataFrame(
        data={
            "a": [
                np.array([1, 2]),
                np.array([3, 4]),
                np.array([5, 6]),
                np.array([7, 8, 9]),
            ],
            "b": [
                np.array([0, 1]),
                np.array([0, 1]),
                np.array([0, 1]),
                np.array([0, 1, 0]),
            ],
        },
        index=[1, 2, 3, 4],
        dtype=pd.ArrowDtype(pa.list_(pa.int64())),
    )
    assert_frame_equal(nested_df, desired_nested)

    assert_array_equal(nested_df.attrs["_offset"], [0, 2, 4, 6, 9])
    assert_array_equal(nested_df.attrs["_offset"], nested_df["a"].attrs["_offset"])
    assert_array_equal(nested_df.attrs["_offset"], nested_df["b"].attrs["_offset"])


def test_view_sorted_series_as_nested_array():
    series = pd.Series(
        data=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        index=[1, 1, 2, 2, 3, 3, 4, 4, 4],
    )
    nested = packer.view_sorted_series_as_nested_array(series)

    assert_array_equal(nested.index, [1, 2, 3, 4])

    desired_nested = pd.Series(
        data=[
            np.array([1, 2]),
            np.array([3, 4]),
            np.array([5, 6]),
            np.array([7, 8, 9]),
        ],
        index=[1, 2, 3, 4],
        dtype=pd.ArrowDtype(pa.list_(pa.int64())),
    )
    assert_series_equal(nested, desired_nested)

    assert_array_equal(nested.attrs["_offset"], [0, 2, 4, 6, 9])


def test_calculate_sorted_index_offsets():
    index = pd.Index([1, 1, 2, 2, 3, 3, 4, 4])
    offset = packer.calculate_sorted_index_offsets(index)
    assert_array_equal(offset, [0, 2, 4, 6, 8])
