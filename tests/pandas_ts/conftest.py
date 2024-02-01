from numpy.testing import assert_array_equal


def assert_nested_array_series_equal(a, b):
    assert_array_equal(a.index, b.index)
    for inner_a, inner_b in zip(a, b):
        assert_array_equal(inner_a, inner_b, err_msg=f"Series '{a.name}' is not equal series '{b.name}'")


def assert_df_equal(a, b):
    assert_array_equal(a.index, b.index)
    assert_array_equal(a.columns, b.columns)
    assert_array_equal(a.dtypes, b.dtypes)
    for column in a.columns:
        assert_array_equal(a[column], b[column], err_msg=f"Column '{column}' is not equal column '{column}'")
