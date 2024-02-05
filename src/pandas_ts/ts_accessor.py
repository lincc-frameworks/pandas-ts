from typing import cast

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api.extensions import register_series_accessor

from pandas_ts.utils import is_pa_type_a_list

__all__ = ["TsAccessor"]


@register_series_accessor("ts")
class TsAccessor:
    """Accessor for operations on Series of TsDtype"""

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
            if not is_pa_type_a_list(field.type):
                raise AttributeError(
                    "Can only use .ts accessor with a Series with dtype pyarrow struct dtype, all fields "
                    f"must be list types. Given struct has unsupported field {field}"
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
