# We do not import pandas-ts directly, because we want to avoid ahead-of-time accessor registration

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal

from pandas_ts import TsDtype


def test_ts_accessor_registered():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([1.0, 2.0, 1.0])]),
            pa.array([np.array([4, 5, 6]), np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[0, 1])

    _accessor = series.ts


def test_ts_accessor_to_lists():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), -np.array([1.0, 2.0, 1.0])]),
            pa.array([np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[0, 1])

    lists = series.ts.to_lists()

    desired = pd.DataFrame(
        data={
            "a": pd.Series(
                data=[np.array([1.0, 2.0, 3.0]), -np.array([1.0, 2.0, 1.0])],
                dtype=pd.ArrowDtype(pa.list_(pa.float64())),
                index=[0, 1],
            ),
            "b": pd.Series(
                data=[np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])],
                dtype=pd.ArrowDtype(pa.list_(pa.float64())),
                index=[0, 1],
            ),
        },
    )
    assert_frame_equal(lists, desired)


def test_ts_accessor_to_lists_with_fields():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), -np.array([1.0, 2.0, 1.0])]),
            pa.array([np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[0, 1])

    lists = series.ts.to_lists(fields=["a"])

    desired = pd.DataFrame(
        data={
            "a": pd.Series(
                data=[np.array([1.0, 2.0, 3.0]), -np.array([1.0, 2.0, 1.0])],
                dtype=pd.ArrowDtype(pa.list_(pa.float64())),
                index=[0, 1],
            ),
        },
    )
    assert_frame_equal(lists, desired)


def test_ts_accessor_to_flat():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[0, 1])

    flat = series.ts.to_flat()

    desired = pd.DataFrame(
        data={
            "a": pd.Series(
                data=[1.0, 2.0, 3.0, 1.0, 2.0, 1.0],
                index=[0, 0, 0, 1, 1, 1],
                name="a",
                copy=False,
            ),
            "b": pd.Series(
                data=[-4.0, -5.0, -6.0, -3.0, -4.0, -5.0],
                index=[0, 0, 0, 1, 1, 1],
                name="b",
                copy=False,
            ),
        },
    )

    assert_array_equal(flat.dtypes, desired.dtypes)
    assert_array_equal(flat.index, desired.index)

    for column in flat.columns:
        assert_array_equal(flat[column], desired[column])


def test_to_flat_with_fields():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[0, 1])

    flat = series.ts.to_flat(fields=["a"])

    desired = pd.DataFrame(
        data={
            "a": pd.Series(
                data=[1.0, 2.0, 3.0, 1.0, 2.0, 1.0],
                index=[0, 0, 0, 1, 1, 1],
                name="a",
                copy=False,
            ),
        },
    )

    assert_array_equal(flat.dtypes, desired.dtypes)
    assert_array_equal(flat.index, desired.index)

    for column in flat.columns:
        assert_array_equal(flat[column], desired[column])


def test_ts_accessor_fields():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[0, 1])

    assert_array_equal(series.ts.fields, ["a", "b"])


def test_ts_accessor_flat_length():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[0, 1])

    assert series.ts.flat_length == 6


def test_set_flat_field():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[0, 1])

    series.ts.set_flat_field("a", np.array(["a", "b", "c", "d", "e", "f"]))

    assert_series_equal(
        series.ts["a"],
        pd.Series(
            data=[["a", "b", "c"], ["d", "e", "f"]],
            index=[0, 1],
            name="a",
            dtype=pd.ArrowDtype(pa.list_(pa.string())),
        ),
    )


def test_set_list_field():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[0, 1])

    series.ts.set_list_field("c", [["a", "b", "c"], ["d", "e", "f"]])

    assert_series_equal(
        series.ts["c"],
        pd.Series(
            data=[["a", "b", "c"], ["d", "e", "f"]],
            index=[0, 1],
            name="c",
            dtype=pd.ArrowDtype(pa.list_(pa.string())),
        ),
    )


def test_delete_field():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[0, 1])

    a = series.ts.delete_field("a")

    assert_array_equal(series.ts.fields, ["b"])
    assert_series_equal(
        a,
        pd.Series(
            [np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])],
            dtype=pd.ArrowDtype(pa.list_(pa.float64())),
            index=[0, 1],
            name="a",
        ),
    )


def test_ts_accessor___getitem__():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[0, 1])

    assert_series_equal(
        series.ts["a"],
        pd.Series(
            [np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])],
            dtype=pd.ArrowDtype(pa.list_(pa.float64())),
            index=[0, 1],
            name="a",
        ),
    )
    assert_series_equal(
        series.ts["b"],
        pd.Series(
            [-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])],
            dtype=pd.ArrowDtype(pa.list_(pa.float64())),
            index=[0, 1],
            name="b",
        ),
    )


def test_ts_accessor___setitem___with_flat():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[0, 1])

    series.ts["a"] = np.array(["a", "b", "c", "d", "e", "f"])

    assert_series_equal(
        series.ts["a"],
        pd.Series(
            data=[["a", "b", "c"], ["d", "e", "f"]],
            index=[0, 1],
            name="a",
            dtype=pd.ArrowDtype(pa.list_(pa.string())),
        ),
    )


def test_ts_accessor___setitem___with_list():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[0, 1])

    series.ts["c"] = [["a", "b", "c"], ["d", "e", "f"]]

    assert_series_equal(
        series.ts["c"],
        pd.Series(
            data=[["a", "b", "c"], ["d", "e", "f"]],
            index=[0, 1],
            name="c",
            dtype=pd.ArrowDtype(pa.list_(pa.string())),
        ),
    )


def test_ts_accessor___setited___raises_for_ambiguous_lengths_1():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array(
                [
                    np.array(
                        [
                            1.0,
                        ]
                    ),
                    np.array([2.0]),
                ]
            ),
            pa.array([-np.array([6.0]), -np.array([5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[0, 1])

    with pytest.raises(ValueError):
        series.ts["c"] = ["a", "b", "c"]


def test_ts_accessor___setited___raises_for_ambiguous_lengths_2():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0]), np.array([])]),
            pa.array([-np.array([6.0, 5.0]), -np.array([])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[0, 1])

    with pytest.raises(ValueError):
        series.ts["c"] = ["a", "b", "c"]


def test_ts_accessor___delitem__():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[0, 1])

    del series.ts["a"]

    assert_array_equal(series.ts.fields, ["b"])


def test_ts_accessor___iter__():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[0, 1])

    assert_array_equal(list(series.ts), ["a", "b"])


def test_ts_accessor___len__():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[0, 1])

    assert len(series.ts) == 2
