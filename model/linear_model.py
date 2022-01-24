from __future__ import annotations

from numpy.typing import ArrayLike
from sklearn.linear_model import LinearRegression

from .regression_model import RegressionModel


class LinearModel(RegressionModel):
    def __init__(self):
        self._lin_regressor: LinearRegression | None = None

    def learn(self, x_train: ArrayLike, y_train: ArrayLike):
        lin_regressor = LinearRegression()
        lin_regressor.fit(x_train, y_train)
        self._lin_regressor = lin_regressor

    def predict(self, x_test: ArrayLike) -> ArrayLike:
        lin_regressor = self._lin_regressor
        return lin_regressor.predict(x_test)
