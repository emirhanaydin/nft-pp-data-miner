from dataclasses import dataclass

from numpy.typing import ArrayLike
from sklearn.model_selection import KFold

from metrics import Metrics
from model import Model


@dataclass
class TestPredictionResult:
    x_test: ArrayLike
    prediction: ArrayLike


@dataclass
class CrossValidationResult:
    best_test_pred: TestPredictionResult
    metrics: Metrics
    test_predictions: list[TestPredictionResult]


class CrossValidator:
    def __init__(self, x: ArrayLike, y: ArrayLike):
        self._x = x
        self._y = y

    def predict(
            self,
            model: Model,
            n_splits=10,
    ) -> CrossValidationResult:
        x = self._x
        y = self._y
        metrics = Metrics.max_error()
        best_test_pred: TestPredictionResult
        test_predictions: list[TestPredictionResult] = []

        kf = KFold(n_splits=n_splits)
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.learn(x_train, y_train)

            y_pred = model.predict(x_test)

            m = Metrics.from_test_pred(y_test, y_pred)
            if m.rmse < metrics.rmse:
                metrics = m
                best_test_pred = TestPredictionResult(x_test, y_pred)

            test_predictions.append(TestPredictionResult(x_test, y_pred))

        # noinspection PyUnboundLocalVariable
        return CrossValidationResult(best_test_pred, metrics, test_predictions)
