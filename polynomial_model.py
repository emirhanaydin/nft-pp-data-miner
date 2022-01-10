from numpy.typing import ArrayLike
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from model import Model


class PolynomialModel(Model):
    def __init__(self, degree: int = 2):
        self._degree = degree
        self._poly: type(PolynomialFeatures) = None
        self._lin_regressor: type(LinearRegression) = None

    def learn(self, x_train: ArrayLike, y_train: ArrayLike):
        degree = self._degree
        poly = PolynomialFeatures(degree=degree)

        x_transform = poly.fit_transform(x_train)
        poly.fit(x_transform, y_train)

        lin_regressor = LinearRegression()
        lin_regressor.fit(x_transform, y_train)

        self._poly = poly
        self._lin_regressor = lin_regressor

    def predict(self, x_test: ArrayLike) -> ArrayLike:
        poly = self._poly
        lin_regressor = self._lin_regressor

        x_test_transform = poly.fit_transform(x_test)
        return lin_regressor.predict(x_test_transform)
