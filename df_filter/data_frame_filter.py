from __future__ import annotations

from typing import Callable

from pandas import DataFrame


class DataFrameFilter:
    def __init__(self, df: DataFrame):
        self._df = df
        self._filter_fns = []

    def pipe(self, filter_fn: Callable[[DataFrame], DataFrame]) -> DataFrameFilter:
        self._filter_fns.append(filter_fn)
        return self

    def filter(self) -> DataFrame:
        df = self._df.copy()
        for fn in self._filter_fns:
            df = fn(df)

        return df
