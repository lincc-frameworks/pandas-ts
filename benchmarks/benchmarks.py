"""Two sample benchmarks to compute runtime and memory usage.

For more information on writing benchmarks:
https://asv.readthedocs.io/en/stable/writing_benchmarks.html."""


import numpy as np
import pandas as pd
from pandas_ts.packer import pack_flat


def dummy_flat_df(n_object, n_per_object):
    n_flat = n_object * n_per_object
    return pd.DataFrame(
        {
            "x": np.arange(n_flat),
            "y": np.linspace(0, 1, n_flat),
        },
        index=np.tile(np.arange(n_object), n_per_object),
    )


class PackAndApplyLength:
    def setup(self):
        self.flat = dummy_flat_df(10_000, 100)

    def time_packed(self):
        series = pack_flat(self.flat)
        _result = series.apply(len)

    def time_groupby(self):
        _result = self.flat.groupby(level=0).apply(len)


class PackAndApplyLengthFiveTimes:
    def setup(self):
        self.flat = dummy_flat_df(10_000, 100)

    def time_packed(self):
        series = pack_flat(self.flat)
        for _ in range(5):
            _result = series.apply(len)

    def time_groupby(self):
        for _ in range(5):
            _result = self.flat.groupby(level=0).apply(len)

    def time_sort_groupby(self):
        sorted = self.flat.sort_index(kind="stable")
        for _ in range(5):
            _result = sorted.groupby(level=0).apply(len)


class Pack:
    def setup(self):
        self.flat = dummy_flat_df(10_000, 100)

    def time_pack_flat(self):
        _result = pack_flat(self.flat)
