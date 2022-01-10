from numpy.typing import ArrayLike
from sklearn.linear_model import LinearRegression

from model import Model


class LinearModel(Model):
    def __init__(self):
        self._lin_regressor: type(LinearRegression) = None

    def learn(self, x_train: ArrayLike, y_train: ArrayLike):
        lin_regressor = LinearRegression()
        lin_regressor.fit(x_train, y_train)
        self._lin_regressor = lin_regressor

    def predict(self, x_test: ArrayLike) -> ArrayLike:
        lin_regressor = self._lin_regressor
        return lin_regressor.predict(x_test)
