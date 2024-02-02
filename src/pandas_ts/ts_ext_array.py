from __future__ import annotations

from typing import Any, Iterator, cast

import numpy as np
import pandas as pd
import pyarrow as pa
from numpy.typing import DTypeLike

# Needed by ArrowExtensionArray.to_numpy(na_value=no_default)
from pandas._libs.lib import no_default

# It is considered to be an experimental, so we need to be careful with it.
from pandas.core.arrays import ArrowExtensionArray
from pyarrow import ExtensionArray

from pandas_ts.ts_dtype import TsDtype
from pandas_ts.utils import is_pa_type_a_list

__all__ = ["TsExtensionArray"]


class TsExtensionArray(ArrowExtensionArray):
    """Pandas extension array for TS structure array

    Parameters
    ----------
    values : pyarrow.Array or pyarrow.ChunkedArray
        The array to be wrapped, must be a struct array with all fields being list
        arrays of the same lengths.

    validate : bool, default True
        Whether to validate the input array.

    Raises
    ------
    ValueError
        If the input array is not a struct array or if any of the fields is not
        a list array or if the list arrays have different lengths.
    """

    _dtype: TsDtype

    def __init__(self, values: pa.Array | pa.ChunkedArray, *, validate: bool = True) -> None:
        super().__init__(values=values)

        # Fix the dtype to be TsDtype
        self._dtype = TsDtype.from_pandas_arrow_dtype(self._dtype)

        if validate:
            self._validate(self._pa_array)

    @staticmethod
    def _validate(array: pa.ChunkedArray) -> None:
        for chunk in array.iterchunks():
            if not pa.types.is_struct(chunk.type):
                raise ValueError(f"Expected a StructArray, got {chunk.type}")
            struct_array = cast(pa.StructArray, chunk)

            first_list_array: pa.ListArray | None = None
            for field in struct_array.type:
                inner_array = struct_array.field(field.name)
                if not is_pa_type_a_list(inner_array.type):
                    raise ValueError(f"Expected a ListArray, got {inner_array.type}")
                list_array = cast(pa.ListArray, inner_array)

                if first_list_array is None:
                    first_list_array = list_array
                    continue
                # compare offsets from the first list array with the current one
                if not first_list_array.offsets.equals(list_array.offsets):
                    raise ValueError("Offsets of all ListArrays must be the same")

    def __getitem__(self, item):
        value = super().__getitem__(item)
        # Convert "scalar" value to pd.DataFrame
        if not isinstance(value, dict):
            return value
        return pd.DataFrame(value, copy=True)

    def __iter__(self) -> Iterator[Any]:
        for value in super().__iter__():
            # Convert "scalar" value to pd.DataFrame
            if not isinstance(value, dict):
                yield value
            else:
                yield pd.DataFrame(value, copy=True)

    def to_numpy(
        self, dtype: DTypeLike | None = None, copy: bool = False, na_value: Any = no_default
    ) -> np.ndarray:
        array = super().to_numpy(dtype=dtype, copy=copy, na_value=na_value)

        # Hack with np.empty is the only way to force numpy to create 1-d array of objects
        result = np.empty(shape=array.shape, dtype=object)
        # We do copy=False here because user's 'copy' is already handled by ArrowExtensionArray.to_numpy
        result[:] = [pd.DataFrame(value, copy=False) for value in array]
        return result

    @property
    def list_offsets(self) -> pa.ChunkedArray:
        """The list offsets of the field arrays.

        It is a chunk array of list offsets of the first field array.
        (All fields must have the same offsets.)
        """
        return pa.chunked_array([chunk.field(0).offsets for chunk in self._pa_array.iterchunks()])
