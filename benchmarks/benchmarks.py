"""Two sample benchmarks to compute runtime and memory usage.

For more information on writing benchmarks:
https://asv.readthedocs.io/en/stable/writing_benchmarks.html."""

from time import sleep


def time_computation():
    """Time computations are prefixed with 'time'."""
    sleep(1e-3)


def mem_list():
    """Memory computations are prefixed with 'mem' or 'peakmem'."""
    sleep(1e-3)
