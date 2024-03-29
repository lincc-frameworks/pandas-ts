# Python 3.9 doesn't support "|" for types
from __future__ import annotations

from collections.abc import Generator, MutableMapping
from typing import cast

import numpy as np
import pandas as pd
import pyarrow as pa
from numpy.typing import ArrayLike
from pandas.api.extensions import register_series_accessor

from pandas_ts.packer import pack_sorted_df_into_struct
from pandas_ts.ts_dtype import TsDtype

__all__ = ["TsAccessor"]


@register_series_accessor("ts")
class TsAccessor(MutableMapping):
    """Accessor for operations on Series of TsDtype

    This accessor implements `MutableMapping` interface over the fields of the
    struct, so you can access, change and delete the fields as if it was a
    dictionary, with `[]`, `[] =` and `del` operators.
    """

    def __init__(self, series):
        self._check_series(series)

        self._series = series

    @staticmethod
    def _check_series(series):
        dtype = series.dtype
        if not isinstance(dtype, TsDtype):
            raise AttributeError(f"Can only use .ts accessor with a Series of TsDtype, got {dtype}")

    def to_lists(self, fields: list[str] | None = None) -> pd.DataFrame:
        """Convert ts into dataframe of list-array columns

        Parameters
        ----------
        fields : list[str] or None, optional
            Names of the fields to include. Default is None, which means all fields.

        Returns
        -------
        pd.DataFrame
            Dataframe of list-arrays.
        """
        df = self._series.struct.explode()
        if fields is None:
            return df
        return df[fields]

    def to_flat(self, fields: list[str] | None = None) -> pd.DataFrame:
        """Convert ts into dataframe of flat arrays

        Parameters
        ----------
        fields : list[str] or None, optional
            Names of the fields to include. Default is None, which means all fields.

        Returns
        -------
        pd.DataFrame
            Dataframe of flat arrays.
        """
        # For some reason, .struct.dtypes is cached, so we will use TsExtensionArray directly
        fields = fields if fields is not None else list(self._series.array.field_names)
        if len(fields) == 0:
            raise ValueError("Cannot flatten a struct with no fields")

        flat_series = {}
        index = None
        for field in fields:
            list_array = cast(pa.ListArray, pa.array(self._series.struct.field(field)))
            if index is None:
                index = np.repeat(self._series.index.values, np.diff(list_array.offsets))
            flat_series[field] = pd.Series(
                list_array.flatten(),
                index=index,
                name=field,
                copy=False,
            )
        return pd.DataFrame(flat_series)

    @property
    def flat_length(self) -> int:
        """Length of the flat arrays"""
        return self._series.array.flat_length

    @property
    def fields(self) -> list[str]:
        """Names of the nested columns"""
        # For some reason, .struct.dtypes is cached, so we will use TsExtensionArray directly
        return self._series.array.field_names

    def set_flat_field(self, field: str, value: ArrayLike) -> None:
        """Set the field from flat-array of values, in-place

        Parameters
        ----------
        field : str
            Name of the field to set. If not present, it will be added.
        value : ArrayLike
            Array of values to set. It must be a scalar or have the same length
             as the flat arrays, e.g. `self.flat_length`.
        """
        self._series.array.set_flat_field(field, value)

    def set_list_field(self, field: str, value: ArrayLike) -> None:
        """Set the field from list-array, in-place

        Parameters
        ----------
        field : str
            Name of the field to set. If not present, it will be added.
        value : ArrayLike
            Array of values to set. It must be a list-array of the same length
             as the series, e.g. length of the series.
        """
        self._series.array.set_list_field(field, value)

    # I intentionally don't call it `drop` or `drop_field` because `pd.DataFrame.drop` is not inplace
    # by default, and I wouldn't like to surprise the user.
    def pop_field(self, field: str) -> pd.Series:
        """Delete the field from the struct and return it.

        Parameters
        ----------
        field : str
            Name of the field to delete.

        Returns
        -------
        pd.Series
            The deleted field.
        """
        series = self[field]
        self._series.array.pop_field(field)
        return series

    def query_flat(self, query: str) -> pd.Series:
        """Query the flat arrays with a boolean expression

        Currently, it will remove empty rows from the output series.
        # TODO: preserve the index keeping empty rows

        Parameters
        ----------
        query : str
            Boolean expression to filter the rows.

        Returns
        -------
        pd.Series
            The filtered series.
        """
        flat = self.to_flat().query(query)
        if len(flat) == 0:
            return pd.Series([], dtype=self._series.dtype)
        return pack_sorted_df_into_struct(flat)

    def get_list_series(self, field: str) -> pd.Series:
        """Get the list-array field as a Series

        Parameters
        ----------
        field : str
            Name of the field to get.

        Returns
        -------
        pd.Series
            The list-array field.
        """
        return self._series.struct.field(field)

    def __getitem__(self, key: str | list[str]) -> pd.Series:
        if isinstance(key, list):
            new_array = self._series.array.view_fields(key)
            return pd.Series(new_array, index=self._series.index, name=self._series.name)

        series = self._series.struct.field(key).list.flatten()
        series.index = np.repeat(self._series.index.values, np.diff(self._series.array.list_offsets))
        series.name = key
        return series

    def __setitem__(self, key: str, value: ArrayLike) -> None:
        # TODO: we can be much-much smarter about the performance here
        # TODO: think better about underlying pa.ChunkArray in both self._series.array and value

        # Everything is empty, do nothing
        if len(self._series) == 0 and np.ndim(value) != 0:
            array = pa.array(value)
            if len(array) == 0:
                return

        if len(self._series) == self.flat_length:
            raise ValueError(
                f"Cannot use `.ts[{key}] = value` when the series has the same count of 'list' rows as 'flat'"
                "rows, because it is ambiguous whether the input is a 'flat' or a 'list' array. Use"
                "`.ts.set_flat_field()` or `.ts.set_list_field()` instead."
            )

        # Set single value for all rows
        if np.ndim(value) == 0:
            self.set_flat_field(key, value)
            return

        pa_array = pa.array(value)

        # Input is a flat array of values
        if len(pa_array) == self.flat_length:
            self.set_flat_field(key, pa_array)
            return

        # Input is a list-array of values
        if len(pa_array) == len(self._series):
            self.set_list_field(key, pa_array)
            return

        raise ValueError(
            f"Cannot set field {key} with value of length {len(pa_array)}, the value is expected to be "
            f"either a scalar, a 'flat' array of length {self.flat_length}, or a 'list' array of length "
            f"{len(self._series)}."
        )

    def __delitem__(self, key: str) -> None:
        self.pop_field(key)

    def __iter__(self) -> Generator[str, None, None]:
        # For some reason, .struct.dtypes is cached, so we will use TsExtensionArray directly
        yield from iter(self._series.array.field_names)

    def __len__(self) -> int:
        # For some reason, .struct.dtypes is cached, so we will use TsExtensionArray directly
        return len(self._series.array.field_names)
