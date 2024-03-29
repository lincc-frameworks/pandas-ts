# We do not import pandas-ts directly, because we want to avoid ahead-of-time accessor registration

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal

from pandas_ts import TsDtype
from pandas_ts.ts_ext_array import TsExtensionArray


def test_registered():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([1.0, 2.0, 1.0])]),
            pa.array([np.array([4, 5, 6]), np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[0, 1])

    _accessor = series.ts


def test_to_lists():
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


def test_to_lists_with_fields():
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


def test_to_flat():
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


def test_fields():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[0, 1])

    assert_array_equal(series.ts.fields, ["a", "b"])


def test_flat_length():
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
            data=["a", "b", "c", "d", "e", "f"],
            index=[0, 0, 0, 1, 1, 1],
            name="a",
            dtype=pd.ArrowDtype(pa.string()),
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
            data=["a", "b", "c", "d", "e", "f"],
            index=[0, 0, 0, 1, 1, 1],
            name="c",
            dtype=pd.ArrowDtype(pa.string()),
        ),
    )


def test_pop_field():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[0, 1])

    a = series.ts.pop_field("a")

    assert_array_equal(series.ts.fields, ["b"])
    assert_series_equal(
        a,
        pd.Series(
            [1.0, 2.0, 3.0, 1.0, 2.0, 1.0],
            dtype=pd.ArrowDtype(pa.float64()),
            index=[0, 0, 0, 1, 1, 1],
            name="a",
        ),
    )


def test_query_flat_1():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]),
            pa.array([np.array([6.0, 4.0, 2.0]), np.array([1.0, 2.0, 3.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[5, 7])

    filtered = series.ts.query_flat("a + b >= 7.0")

    desired_struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0]), np.array([5.0, 6.0])]),
            pa.array([np.array([6.0]), np.array([2.0, 3.0])]),
        ],
        names=["a", "b"],
    )
    desired = pd.Series(desired_struct_array, dtype=TsDtype(desired_struct_array.type), index=[5, 7])

    assert_series_equal(filtered, desired)


# Currently we remove empty rows from the output series
def test_query_flat_2():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]),
            pa.array([np.array([6.0, 4.0, 2.0]), np.array([1.0, 2.0, 3.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[5, 7])

    filtered = series.ts.query_flat("a > 1000.0")
    desired = pd.Series([], dtype=series.dtype)

    assert_series_equal(filtered, desired)


def test_get_list_series():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([4, 5, 6])]),
            pa.array([np.array([6, 4, 2]), np.array([1, 2, 3])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[5, 7])

    lists = series.ts.get_list_series("a")

    assert_series_equal(
        lists,
        pd.Series(
            data=[np.array([1, 2, 3]), np.array([4, 5, 6])],
            dtype=pd.ArrowDtype(pa.list_(pa.int64())),
            index=[5, 7],
            name="a",
        ),
    )


def test___getitem___single_field():
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
            np.array([1.0, 2.0, 3.0, 1.0, 2.0, 1.0]),
            dtype=pd.ArrowDtype(pa.float64()),
            index=[0, 0, 0, 1, 1, 1],
            name="a",
        ),
    )
    assert_series_equal(
        series.ts["b"],
        pd.Series(
            -np.array([4.0, 5.0, 6.0, 3.0, 4.0, 5.0]),
            dtype=pd.ArrowDtype(pa.float64()),
            index=[0, 0, 0, 1, 1, 1],
            name="b",
        ),
    )


def test___getitem___multiple_fields():
    arrays = [
        pa.array([np.array([1.0, 2.0, 3.0]), -np.array([1.0, 2.0, 1.0])]),
        pa.array([np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
    ]
    series = pd.Series(
        TsExtensionArray(
            pa.StructArray.from_arrays(
                arrays=arrays,
                names=["a", "b"],
            )
        ),
        index=[0, 1],
    )

    assert_series_equal(
        series.ts[["b", "a"]],
        pd.Series(
            TsExtensionArray(
                pa.StructArray.from_arrays(
                    arrays=arrays[::-1],
                    names=["b", "a"],
                )
            ),
            index=[0, 1],
        ),
    )


def test___setitem___with_flat():
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
            data=["a", "b", "c", "d", "e", "f"],
            index=[0, 0, 0, 1, 1, 1],
            name="a",
            dtype=pd.ArrowDtype(pa.string()),
        ),
    )


def test___setitem___with_list():
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
            data=["a", "b", "c", "d", "e", "f"],
            index=[0, 0, 0, 1, 1, 1],
            name="c",
            dtype=pd.ArrowDtype(pa.string()),
        ),
    )


def test___setited___raises_for_ambiguous_lengths_1():
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


def test___setited___raises_for_ambiguous_lengths_2():
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


def test___delitem__():
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


def test___iter__():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[0, 1])

    assert_array_equal(list(series.ts), ["a", "b"])


def test___len__():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=TsDtype(struct_array.type), index=[0, 1])

    assert len(series.ts) == 2
