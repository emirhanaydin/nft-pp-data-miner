import sys

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import mean_squared_error, r2_score


class Metrics:
    def __init__(self, mse: float, rmse: float, r2: float):
        self._mse = mse
        self._rmse = rmse
        self._r2 = r2

    @staticmethod
    def from_test_pred(y_test: ArrayLike, y_pred: ArrayLike):
        mse: float = mean_squared_error(y_test, y_pred)
        rmse: float = np.sqrt(mse)
        r2: float = r2_score(y_test, y_pred)
        return Metrics(mse=mse, rmse=rmse, r2=r2)

    @staticmethod
    def max_error():
        float_info = sys.float_info
        return Metrics(
            mse=float_info.max,
            rmse=float_info.max,
            r2=float_info.min
        )

    @property
    def mse(self):
        return self._mse

    @property
    def rmse(self):
        return self._rmse

    @property
    def r2(self):
        return self._r2

    def __str__(self) -> str:
        return '\n'.join([
            f'MSE: {self._mse}',
            f'RMSE: {self._rmse}',
            f'R^2: {self.r2}',
        ])
