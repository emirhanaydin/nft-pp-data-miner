from abc import ABC, abstractmethod

from numpy.typing import ArrayLike


class Model(ABC):
    @abstractmethod
    def learn(self, x_train: ArrayLike, y_train: ArrayLike):
        pass

    @abstractmethod
    def predict(self, x_test: ArrayLike) -> ArrayLike:
        pass
