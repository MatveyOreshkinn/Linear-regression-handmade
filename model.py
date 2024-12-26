import numpy as np
import pandas as pd
from typing import Optional, Union, NoReturn


class MyLineReg:
    def __init__(self, weights: Optional[np.ndarray] = None, n_iter: int = 100, learning_rate: float = 0.1) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

    def __str__(self):
        """
        Возвращает строковое представление объекта MyLineReg.

        Returns:
            str: Строковое представление объекта.

        """
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool = False) -> None:
        """
        Метод fit для обучения модели.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Матрица признаков.
            y (Union[pd.Series, np.ndarray]): Вектор целевых значений.
            verbose (Union[bool, int], optional): Флаг для включения детального вывода. Defaults to False.

        """
        X.insert(0, 'x0', 1)  # оптимизация для нахождения градиента
        cnt_features = X.shape[1]

        if self.weights is None:
            self.weights = np.ones(cnt_features)

        loss = np.mean((X.dot(self.weights) - y) ** 2)
        if verbose:
            print(f'start | loss: {loss}')

        for i in range(self.n_iter):
            y_pred = np.dot(X, self.weights)
            loss = np.mean((y_pred - y) ** 2)

            if verbose and i % verbose == 0:
                print(f'{i} | loss: {loss}')

            grad = (2 / X.shape[0]) * np.dot((y_pred - y), X)
            self.weights -= self.learning_rate * grad

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Метод predict для предсказания значений.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Матрица признаков.

        Returns:
             np.ndarray: Вектор предсказанных значений.

        """
        X.insert(0, 'x0', 1)
        return np.dot(X, self.weights)

    def get_coef(self) -> np.ndarray:
        """
        Возвращает коэффициенты модели (без bias).

        Returns:
           np.ndarray: Вектор коэффициентов модели.

        """
        return self.weights[1:]
