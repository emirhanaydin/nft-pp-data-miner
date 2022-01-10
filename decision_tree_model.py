from numpy.typing import ArrayLike
from sklearn.tree import DecisionTreeRegressor

from model import Model


class DecisionTreeModel(Model):
    def __init__(self, max_depth: int = None):
        self._max_depth = max_depth
        self._tree_regressor: type(DecisionTreeRegressor) = None

    def learn(self, x_train: ArrayLike, y_train: ArrayLike):
        max_depth = self._max_depth
        tree_regressor = DecisionTreeRegressor(max_depth=max_depth)
        tree_regressor.fit(x_train, y_train)
        self._tree_regressor = tree_regressor

    def predict(self, x_test: ArrayLike) -> ArrayLike:
        tree_regressor = self._tree_regressor
        return tree_regressor.predict(x_test)
