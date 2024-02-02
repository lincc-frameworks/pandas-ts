import pyarrow as pa


def is_pa_type_a_list(pa_type):
    return (
        pa.types.is_list(pa_type) or pa.types.is_large_list(pa_type) or pa.types.is_fixed_size_list(pa_type)
    )
