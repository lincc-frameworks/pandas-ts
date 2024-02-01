from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api.extensions import register_series_accessor

__all__ = ["TsAccessor"]


def pa_type_is_any_list(pa_type):
    return (
        pa.types.is_list(pa_type) or pa.types.is_large_list(pa_type) or pa.types.is_fixed_size_list(pa_type)
    )


@register_series_accessor("ts")
class TsAccessor:
    def __init__(self, series):
        self._check_series(series)

        self._series = series

    @staticmethod
    def _check_series(series):
        dtype = series.dtype
        TsAccessor._check_dtype(dtype)

    @staticmethod
    def _check_dtype(dtype):
        # TODO: check if dtype is TsDtype when it is implemented
        if not hasattr(dtype, "pyarrow_dtype"):
            raise AttributeError("Can only use .ts accessor with a Series with dtype pyarrow struct dtype")
        pyarrow_dtype = dtype.pyarrow_dtype
        if not pa.types.is_struct(pyarrow_dtype):
            raise AttributeError("Can only use .ts accessor with a Series with dtype pyarrow struct dtype")

        for field in pyarrow_dtype:
            if not pa_type_is_any_list(field.type):
                raise AttributeError(
                    f"Can only use .ts accessor with a Series with dtype pyarrow struct dtype, all fields must be list types. Given struct has unsupported field {field}"
                )

    def to_nested(self):
        """Convert ts into dataframe of nested arrays"""
        return self._series.struct.explode()

    def to_flat(self):
        """Convert ts into dataframe of flat arrays"""
        fields = self._series.struct.dtypes.index
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
    def fields(self) -> pd.Index:
        """Names of the nested columns"""
        return self._series.struct.dtypes.index

    def __getitem__(self, key: str) -> pd.Series:
        return self._series.struct.field(key)

    def get(self, index: Any) -> pd.DataFrame:
        """Get a single ts item by label (index value) as a dataframe

        Parameters
        ----------
        index : Any
            The label of the item to get, must be in the index of
            the series.

        Returns
        -------
        pd.DataFrame
            A dataframe with the nested arrays of the item.

        See Also
        --------
        pandas_ts.TsAccessor.iget : Get a single ts item by position.
        """
        item = self._series.loc[index]
        return pd.DataFrame.from_dict(item)

    def iget(self, index: int) -> pd.DataFrame:
        """Get a single ts item by position as a dataframe

        Parameters
        ----------
        index : int
            The position of the item to get, must be a valid position
            in the series, i.e. between 0 and len(series) - 1.

        Returns
        -------
        pd.DataFrame
            A dataframe with the nested arrays of the item.

        See Also
        --------
        pandas_ts.TsAccessor.get : Get a single ts item by label (index value).
        """
        item = self._series.iloc[index]
        print(item)
        return pd.DataFrame.from_dict(item)
