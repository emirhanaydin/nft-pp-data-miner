from dataclasses import dataclass

from numpy.typing import ArrayLike
from sklearn.model_selection import KFold

from metrics import Metrics
from model import RegressionModel


@dataclass
class CrossValidationResult:
    prediction: ArrayLike
    metrics: list[Metrics]


class CrossValidator:
    def __init__(self, x: ArrayLike, y: ArrayLike):
        self._x = x
        self._y = y

    def predict(
            self,
            model: RegressionModel,
            n_splits=10,
    ) -> CrossValidationResult:
        x = self._x
        y = self._y
        metrics = Metrics.max_error()
        all_metrics = []
        train_result_index: ArrayLike
        test_result_index: ArrayLike

        kf = KFold(n_splits=n_splits)
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.learn(x_train, y_train)

            y_pred = model.predict(x_test)

            m = Metrics.from_test_pred(y_test, y_pred)
            all_metrics.append(m)

            if m.rmse < metrics.rmse:
                metrics = m
                train_result_index = train_index

        # noinspection PyUnboundLocalVariable
        model.learn(x[train_result_index], y[train_result_index])
        prediction = model.predict(x)

        return CrossValidationResult(prediction, all_metrics)
