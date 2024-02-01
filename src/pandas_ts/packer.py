"""Exploratory module for nested-array data representation conversions

TODO: mask support
TODO: multi-index support
"""

# "|" for python 3.9
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa

__all__ = ["pack_df_into_structs"]


def pack_df(df: pd.DataFrame, name=None) -> pd.DataFrame:
    """Pack a "flat" dataframe into a "nested" dataframe.

    For the input dataframe with repeated indexes, make a pandas.DataFrame,
    where each original column is replaced by a column of lists, and,
    optionally, a "structure" column is added, containing a structure of
    lists with the original columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe, with repeated indexes.

    name : str, optional
        Name of the structure column. The default is None, which means no
        structure column is added.

    Returns
    -------
    pd.DataFrame
        Output dataframe.
    """
    # TODO: we can optimize name=None case a bit
    struct_series = pack_df_into_structs(df)
    packed_df = struct_series.struct.explode()
    if name is not None:
        packed_df[name] = struct_series
    return packed_df


def pack_df_into_structs(df: pd.DataFrame) -> pd.Series:
    """Make a structure of lists representation of a "flat" dataframe.

    For the input dataframe with repeated indexes, make a pandas.Series
    of arrow structures. Each item is a structure of lists, where each
    list contains the values of the corresponding column. The index of
    the output series is the unique index of the input dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe, with repeated indexes.

    Returns
    -------
    pd.Series
        Output series, with unique indexes.
    """

    # TODO: think about the case when the data is pre-sorted and we don't need a data copy.
    flat = df.sort_index()
    return pack_sorted_df_into_struct(flat)


def pack_sorted_df_into_struct(df: pd.DataFrame) -> pd.Series:
    """Make a structure of lists representation of a "flat" dataframe.

    Input dataframe must be sorted and all the columns must have pyarrow dtypes.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe, with repeated indexes. It must be sorted and
        all the columns must have pyarrow dtypes.

    Returns
    -------
    pd.Series
        Output series, with unique indexes.
    """
    packed_df = view_sorted_df_as_nested_arrays(df)
    return view_packed_df_as_struct_series(packed_df)


def view_packed_df_as_struct_series(df: pd.DataFrame) -> pd.Series:
    """Make a series of arrow structures from a dataframe with nested arrays.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe, with nested arrays.

    Returns
    -------
    pd.Series
        Output series, with unique indexes.
    """
    struct_array = pa.StructArray.from_arrays(
        [df[column] for column in df.columns],
        names=df.columns,
    )
    series = pd.Series(
        struct_array,
        dtype=pd.ArrowDtype(struct_array.type),
        index=df.index,
        copy=False,
    )
    if "_offset" in df.attrs:
        series.attrs["_offset"] = df.attrs["_offset"]
    else:
        # TODO: get offsets from underlying pyarrow arrays using any column
        series.attrs["_offset"] = calculate_sorted_index_offsets(df.index)
    return series


def view_sorted_df_as_nested_arrays(df: pd.DataFrame) -> pd.DataFrame:
    """Make a nested array representation of a "flat" dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe, with repeated indexes. It must be sorted by its index.

    Returns
    -------
    pd.DataFrame
        Output dataframe, with unique indexes. It is a view over the input
        dataframe, so it would mute the input dataframe if modified.
        The dataframe and all its columns have an attribute `.attrs["_offset"]`
        with the offsets of the input dataframe index.
    """
    offset_array = calculate_sorted_index_offsets(df.index)
    unique_index = df.index.values[offset_array[:-1]]

    series_ = {
        column: view_sorted_series_as_nested_array(df[column], offset_array, unique_index)
        for column in df.columns
    }

    df = pd.DataFrame(series_)
    df.attrs["_offset"] = offset_array

    return df


def view_sorted_series_as_nested_array(
    series: pd.Series, offset: np.ndarray | None = None, unique_index: np.ndarray | None = None
) -> pd.Series:
    """Make a nested array representation of a "flat" series.

    Parameters
    ----------
    series : pd.Series
        Input series, with repeated indexes. It must be sorted by its index.

    offset: np.ndarray or None, optional
        Pre-calculated offsets of the input series index.
    unique_index: np.ndarray or None, optional
        Pre-calculated unique index of the input series. If given it must be
        equal to `series.index.unique()` and `series.index.values[offset[:-1]]`.

    Returns
    -------
    pd.Series
        Output series, with unique indexes. It is a view over the input series,
        so it would mute the input series if modified. It has an attribute
        `.attrs["_offset"]` with the offsets of the input series index.
    """
    if offset is None:
        offset = calculate_sorted_index_offsets(series.index)
    if unique_index is None:
        unique_index = series.index.values[offset[:-1]]

    list_array = pa.ListArray.from_arrays(
        offset,
        pa.array(series),
    )
    new_series = pd.Series(
        list_array,
        dtype=pd.ArrowDtype(list_array.type),
        index=unique_index,
        copy=False,
    )

    new_series.attrs["_offset"] = offset

    return new_series


def calculate_sorted_index_offsets(index: pd.Index) -> np.ndarray:
    """Calculate the offsets of the pre-sorted index values.

    Parameters
    ----------
    index : pd.Index
        Input index, must be sorted.

    Returns
    -------
    np.ndarray
        Output array of offsets, one element more than the number of unique
        index values.
    """
    # TODO: implement multi-index support
    index_diff = np.diff(index.values, prepend=index.values[0] - 1, append=index.values[-1] + 1)

    if np.any(index_diff < 0):
        raise ValueError("Table index must be strictly sorted.")

    offset = np.nonzero(index_diff)[0]

    return offset
