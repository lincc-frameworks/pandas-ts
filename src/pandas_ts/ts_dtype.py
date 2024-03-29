# Use Self, which is not available until Python 3.11
from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pandas as pd
import pyarrow as pa
from pandas import ArrowDtype
from pandas.api.extensions import register_extension_dtype
from pandas.core.arrays import ArrowExtensionArray

from pandas_ts.utils import is_pa_type_a_list

__all__ = ["TsDtype"]


@register_extension_dtype
class TsDtype(ArrowDtype):
    """Data type to handle packed time series data"""

    pyarrow_dtype: pa.StructType

    def __init__(self, pyarrow_dtype: pa.DataType) -> None:
        pyarrow_dtype = self._validate_dtype(pyarrow_dtype)
        super().__init__(pyarrow_dtype=pyarrow_dtype)

    @classmethod
    def from_fields(cls, fields: Mapping[str, pa.DataType]) -> Self:  # type: ignore[name-defined] # noqa: F821
        """Make TsDtype from a mapping of field names and list item types.

        Parameters
        ----------
        fields : Mapping[str, pa.DataType]
            A mapping of field names and their item types. Since all fields are lists, the item types are
            inner types of the lists, not the list types themselves.

        Returns
        -------
        TsDtype
            The constructed TsDtype.

        Examples
        --------
        >>> dtype = TsDtype.from_fields({"a": pa.float64(), "b": pa.int64()})
        >>> dtype
        ts<a: [double], b: [int64]>
        >>> assert (
        ...     dtype.pyarrow_dtype
        ...     == pa.struct({"a": pa.list_(pa.float64()), "b": pa.list_(pa.int64())})
        ... )
        """
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
                    "TsDtype can only be constructed with pyarrow struct type, all fields must be list "
                    f"type. Given struct has unsupported field {field}"
                )
        return pyarrow_dtype

    @classmethod
    def construct_from_string(cls, string: str) -> Self:  # type: ignore[name-defined] # noqa: F821
        """Construct TsDtype from a string representation.

        This works only for simple types, i.e. non-parametric pyarrow types.

        Parameters
        ----------
        string : str
            The string representation of the ts type. For example,
            'ts<x: [int64], y: [float64]'. It must be consistent with
            the string representation of the dtype given by the `name`
            attribute.

        Returns
        -------
        TsDtype
            The constructed TsDtype.

        Raises
        ------
        ValueError
            If the string is not a valid ts type string or if the element types
            are parametric pyarrow types.
        """
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
            except ValueError as e:
                raise ValueError(
                    "Not a valid ts type string, expected 'ts<x: [type], ...>', got invalid field string "
                    f"'{field_string}'"
                ) from e
            if not field_type.startswith("[") or not field_type.endswith("]"):
                raise ValueError(
                    "Not a valid ts type string, expected 'ts<x: [type], ...>', got invalid field type "
                    f"string '{field_type}'"
                )

            value_type = field_type.removeprefix("[").removesuffix("]")
            # We follow ArrowDtype implementation heere and do not try to parse complex types
            try:
                pa_value_type = pa.type_for_alias(value_type)
            except ValueError as e:
                raise ValueError(
                    f"Parsing pyarrow specific parameters in the string is not supported yet: {value_type}. "
                    "Please use TsDtype() or TsDtype.from_fields() instead."
                ) from e

            fields[field_name] = pa_value_type

        return cls.from_fields(fields)

    @classmethod
    def from_pandas_arrow_dtype(cls, pandas_arrow_dtype: ArrowDtype):
        """Construct TsDtype from a pandas.ArrowDtype.

        Parameters
        ----------
        pandas_arrow_dtype : ArrowDtype
            The pandas.ArrowDtype to construct TsDtype from.

        Returns
        -------
        TsDtype
            The constructed TsDtype.

        Raises
        ------
        ValueError
            If the given dtype is not a valid ts type.
        """
        pyarrow_dtype = cls._validate_dtype(pandas_arrow_dtype.pyarrow_dtype)
        return cls(pyarrow_dtype=pyarrow_dtype)

    @classmethod
    def construct_array_type(cls) -> type[ArrowExtensionArray]:
        """Corresponded array type, always TsExtensionArray"""
        from pandas_ts.ts_ext_array import TsExtensionArray

        return TsExtensionArray

    @property
    def name(self) -> str:
        """The string representation of the ts type"""
        fields = ", ".join([f"{field.name}: [{field.type.value_type!s}]" for field in self.pyarrow_dtype])
        return f"ts<{fields}>"

    type = pd.DataFrame
    """The type of the array's elements, always pd.DataFrame"""
