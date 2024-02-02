# Use Self, which is not available until Python 3.11
from __future__ import annotations

from typing import Mapping, cast

import pandas as pd
import pyarrow as pa
from pandas import ArrowDtype
from pandas.api.extensions import register_extension_dtype
from pandas.core.arrays import ArrowExtensionArray

from pandas_ts.utils import is_pa_type_a_list

__all__ = ["TsDtype"]


@register_extension_dtype
class TsDtype(ArrowDtype):
    pyarrow_dtype: pa.StructType

    def __init__(self, pyarrow_dtype: pa.DataType) -> None:
        pyarrow_dtype = self._validate_dtype(pyarrow_dtype)
        super().__init__(pyarrow_dtype=pyarrow_dtype)

    @classmethod
    def from_fields(cls, fields: Mapping[str, pa.DataType]) -> Self:  # type: ignore[name-defined]
        pyarrow_dtype = pa.struct({field: pa.list_(pa_type) for field, pa_type in fields.items()})
        pyarrow_dtype = cast(pa.StructType, pyarrow_dtype)
        return cls(pyarrow_dtype=pyarrow_dtype)

    @staticmethod
    def _validate_dtype(pyarrow_dtype: pa.DataType) -> pa.StructType:
        if not isinstance(pyarrow_dtype, pa.DataType):
            raise TypeError(f"Expected a 'pyarrow.DataType' object, got {type(pyarrow_dtype)}")
        if not pa.types.is_struct(pyarrow_dtype):
            raise ValueError("TsDtype can only be constructed with pyarrow struct type.")
        pyarrow_dtype = cast(pa.StructType, pyarrow_dtype)

        for field in pyarrow_dtype:
            if not is_pa_type_a_list(field.type):
                raise ValueError(
                    f"TsDtype can only be constructed with pyarrow struct type, all fields must be list types. Given "
                    f"struct has unsupported field {field}"
                )
        return pyarrow_dtype

    @classmethod
    def construct_from_string(cls, string: str) -> Self:  # type: ignore[name-defined]
        if not string.startswith("ts<") or not string.endswith(">"):
            raise ValueError("Not a valid ts type string, expected 'ts<...>'")
        fields_str = string.removeprefix("ts<").removesuffix(">")

        field_strings = fields_str.split(", ")
        if len(field_strings) == 0:
            raise ValueError(
                "Not a valid ts type string, expected at least a single field inside 'ts<x: [type], ...>'"
            )

        fields = {}
        for field_string in field_strings:
            try:
                field_name, field_type = field_string.split(": ", maxsplit=1)
            except ValueError:
                raise ValueError(
                    "Not a valid ts type string, expected 'ts<x: [type], ...>', got invalid field string "
                    f"'{field_string}'"
                )
            if not field_type.startswith("[") or not field_type.endswith("]"):
                raise ValueError(
                    "Not a valid ts type string, expected 'ts<x: [type], ...>', got invalid field type string "
                    f"'{field_type}'"
                )

            value_type = field_type.removeprefix("[").removesuffix("]")
            # We follow ArrowDtype implementation heere and do not try to parse complex types
            try:
                pa_value_type = pa.type_for_alias(value_type)
            except ValueError as e:
                raise ValueError(
                    f"Parsing pyarrow specific parameters in the string is not supported yet: {value_type}. Please use "
                    "TsDtype() or TsDtype.from_fields() instead."
                ) from e

            fields[field_name] = pa_value_type

        return cls.from_fields(fields)

    @classmethod
    def from_pandas_arrow_dtype(cls, pandas_arrow_dtype: ArrowDtype):
        pyarrow_dtype = cls._validate_dtype(pandas_arrow_dtype.pyarrow_dtype)
        return cls(pyarrow_dtype=pyarrow_dtype)

    @classmethod
    def construct_array_type(cls) -> type[ArrowExtensionArray]:
        from pandas_ts.ts_ext_array import TsExtensionArray

        return TsExtensionArray

    @property
    def name(self) -> str:
        fields = ", ".join([f"{field.name}: [{field.type.value_type!s}]" for field in self.pyarrow_dtype])
        return f"ts<{fields}>"

    type = pd.DataFrame
