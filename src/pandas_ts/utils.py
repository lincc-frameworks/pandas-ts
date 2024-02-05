import pyarrow as pa


def is_pa_type_a_list(pa_type: type[pa.Array]) -> bool:
    """Check if the given pyarrow type is a list type.

    I.e. one of the following types: ListArray, LargeListArray,
    FixedSizeListArray.

    Returns
    -------
    bool
        True if the given type is a list type, False otherwise.
    """
    return (
        pa.types.is_list(pa_type) or pa.types.is_large_list(pa_type) or pa.types.is_fixed_size_list(pa_type)
    )
