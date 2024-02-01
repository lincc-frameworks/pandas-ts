from __future__ import annotations

from typing import cast

import pyarrow as pa

# It is considered to be an experimental, so we need to be careful with it.
from pandas.core.arrays import ArrowExtensionArray

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
