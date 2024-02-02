# We do not import pandas-ts directly, because we want to avoid ahead-of-time accessor registration

import importlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from conftest import assert_df_equal, assert_nested_array_series_equal
from numpy.testing import assert_array_equal


@pytest.mark.skip("It is really tricky to test it: some other test may import pandas-ts before this one")
def test_ts_accessor_not_registered():
    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([1.0, 2.0, 1.0])]),
            pa.array([np.array([4, 5, 6]), np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=pd.ArrowDtype(struct_array.type), index=[0, 1])

    with pytest.raises(AttributeError):
        _accessor = series.ts


def test_ts_accessor_registered():
    import pandas_ts as _

    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([1.0, 2.0, 1.0])]),
            pa.array([np.array([4, 5, 6]), np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=pd.ArrowDtype(struct_array.type), index=[0, 1])

    _accessor = series.ts


def test_no_ts_accessor_for_non_struct():
    import pandas_ts as _

    array = pa.array([np.array([1, 2, 3]), np.array([4, 5, 6])])
    series = pd.Series(array, dtype=pd.ArrowDtype(array.type), index=[0, 1])

    with pytest.raises(AttributeError):
        _accessor = series.ts


def test_no_ts_accessor_for_wrong_struct_fields():
    import pandas_ts as _

    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1, 2, 3]), np.array([1.0, 2.0, 1.0])]),
            pa.array([np.array([4, 5, 6]), np.array([3.0, 4.0, 5.0])]),
            pa.array([-6, -7]),
        ],
        names=["a", "b", "c"],
    )
    series = pd.Series(struct_array, dtype=pd.ArrowDtype(struct_array.type), index=[0, 1])

    with pytest.raises(AttributeError):
        _accessor = series.ts


def test_ts_accessor_to_nested():
    import pandas_ts as _

    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), -np.array([1.0, 2.0, 1.0])]),
            pa.array([np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=pd.ArrowDtype(struct_array.type), index=[0, 1])

    nested = series.ts.to_nested()

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

    assert_array_equal(nested.dtypes, desired.dtypes)
    assert_array_equal(nested.index, desired.index)

    for column in nested.columns:
        assert_nested_array_series_equal(nested[column], desired[column])


def test_ts_accessor_to_flat():
    import pandas_ts as _

    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=pd.ArrowDtype(struct_array.type), index=[0, 1])

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


def test_ts_accessor_fields():
    import pandas_ts as _

    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=pd.ArrowDtype(struct_array.type), index=[0, 1])

    assert_array_equal(series.ts.fields, pd.Index(["a", "b"]))


def test_ts_accessor___getitem__():
    import pandas_ts as _

    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=pd.ArrowDtype(struct_array.type), index=[0, 1])

    assert_nested_array_series_equal(
        series.ts["a"],
        pd.Series(
            [np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])],
            dtype=pd.ArrowDtype(pa.list_(pa.float64())),
            index=[0, 1],
        ),
    )
    assert_nested_array_series_equal(
        series.ts["b"],
        pd.Series(
            [-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])],
            dtype=pd.ArrowDtype(pa.list_(pa.float64())),
            index=[0, 1],
        ),
    )


def test_ts_accessor_get():
    import pandas_ts as _

    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )
    series = pd.Series(struct_array, dtype=pd.ArrowDtype(struct_array.type), index=[100, 101])

    second_row_as_df = series.ts.get(101)
    assert_df_equal(
        second_row_as_df, pd.DataFrame({"a": np.array([1.0, 2.0, 1.0]), "b": -np.array([3.0, 4.0, 5.0])})
    )


def test_ts_accessor_iget():
    import pandas_ts as _

    struct_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array([np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])]),
            pa.array([-np.array([4.0, 5.0, 6.0]), -np.array([3.0, 4.0, 5.0])]),
        ],
        names=["a", "b"],
    )

    series = pd.Series(struct_array, dtype=pd.ArrowDtype(struct_array.type), index=[100, 101])

    first_row_as_df = series.ts.iget(-2)
    assert_df_equal(
        first_row_as_df, pd.DataFrame({"a": np.array([1.0, 2.0, 3.0]), "b": -np.array([4.0, 5.0, 6.0])})
    )