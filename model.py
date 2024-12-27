import numpy as np
import pandas as pd
from typing import Optional, Union, NoReturn


class MyLineReg:
    def __init__(self, weights: Optional[np.ndarray] = None, n_iter: int = 100,
                 learning_rate: float = 0.1, metric: Optional[str] = None,
                 reg: Optional[str] = None, l1_coef: float = 0, l2_coef: float = 0) -> None:
        """
        Инициализация класса MyLineReg.

        Args:
            weights (Optional[np.ndarray], optional): Начальные веса модели. По умолчанию None.
            n_iter (int, optional): Количество итераций градиентного спуска. По умолчанию 100.
            learning_rate (float, optional): Скорость обучения. По умолчанию 0.1.
            metric (Optional[str], optional): Метрика для оценки (например, 'mae', 'mse'). По умолчанию None.
            reg (Optional[str], optional): Метод регуляризации ('l1', 'l2' или None). По умолчанию None.
            l1_coef (float, optional): Коэффициент для L1-регуляризации. По умолчанию 0.
            l2_coef (float, optional): Коэффициент для L2-регуляризации. По умолчанию 0.

        """
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.best_score = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

    def __str__(self) -> str:
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
            X (pd.DataFrame): Матрица признаков.
            y (pd.Series): Вектор целевых значений.
            verbose (bool): Флаг для включения детального вывода. Defaults to False.

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

            regular = 0
            if self.reg is not None:
                if self.reg == 'l1':
                    regular = self.l1()
                elif self.reg == 'l2':
                    regular = self.l2()
                else:
                    regular = self.elasticnet()

            grad = (2 / X.shape[0]) * np.dot((y_pred - y), X) + regular
            self.weights -= self.learning_rate * grad

            metric_value = None
            if self.metric is not None:
                if self.metric == 'mae':
                    metric_value = self.mae(X, y)
                elif self.metric == 'mse':
                    metric_value = self.mse(X, y)
                elif self.metric == 'rmse':
                    metric_value = self.rmse(X, y)
                elif self.metric == 'mape':
                    metric_value = self.mape(X, y)
                elif self.metric == 'r2':
                    metric_value = self.r2(X, y)

                self.best_score = metric_value

            if verbose and i % verbose == 0:
                if metric_value is not None:
                    print(
                        f"{i} | loss: {loss:.2f} | {self.metric}: {metric_value:.2f}")
                else:
                    print(f"{i} | loss: {loss:.2f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Метод predict для предсказания значений.

        Args:
            X (pd.DataFrame): Матрица признаков.

        Returns:
             np.ndarray: Вектор предсказанных значений.

        """
        X.insert(0, 'x0', 1)
        return np.dot(X, self.weights)

    def l1(self) -> np.ndarray:
        """
        Вычисляет L1-регуляризацию для текущих весов модели.

        Returns:
            np.ndarray: Значения L1-регуляризации, умноженные на коэффициент l1_coef.

        """
        return self.l1_coef * np.sign(self.weights)

    def l2(self) -> np.ndarray:
        """
        Вычисляет L2-регуляризацию для текущих весов модели.

        Returns:
            np.ndarray: Значения L2-регуляризации, умноженные на коэффициент l2_coef.

        """
        return self.l2_coef * 2 * self.weights

    def elasticnet(self):
        """
        Вычисляет комбинацию L1 и L2-регуляризации (Elastic Net) для текущих весов модели.

        Returns:
            np.ndarray: Сумма L1 и L2-регуляризаций.

        """
        return self.l1() + self.l2()

    def mae(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Вычисляет среднюю абсолютную ошибку (MAE).

        Args:
            X: Матрица признаков (pd.DataFrame).
            y: Вектор истинных значений (pd.Seriesy).

        Returns:
            Средняя абсолютная ошибка (float).

        """
        y_pred = np.dot(X, self.weights)
        return np.mean(abs(y_pred - y))

    def mse(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Вычисляет среднюю квадратичную ошибку (MSE).

        Args:
            X: Матрица признаков (pd.DataFrame).
            y: Вектор истинных значений (pd.Series).

        Returns:
            Средняя квадратичная ошибка (float).

        """
        y_pred = np.dot(X, self.weights)
        return np.mean((y_pred - y) ** 2)

    def mape(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Вычисляет среднюю абсолютную процентную ошибку (MAPE).

        Args:
            X: Матрица признаков (pd.DataFrame).
            y: Вектор истинных значений (pd.Series).

        Returns:
            Средняя абсолютная процентная ошибка (float) в процентах.

        """
        y_pred = np.dot(X, self.weights)
        return np.mean(abs((y - y_pred) / y)) * 100

    def rmse(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Вычисляет корень из средней квадратичной ошибки (RMSE).

        Args:
            X: Матрица признаков (pd.DataFrame).
            y: Вектор истинных значений (pd.Series).

        Returns:
            Корень из средней квадратичной ошибки (float).

        """
        return np.sqrt(self.mse(X, y))

    def r2(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Вычисляет коэффициент детерминации (R^2).

        Args:
            X: Матрица признаков (pd.DataFrame).
            y: Вектор истинных значений (pd.Series).

        Returns:
            Коэффициент детерминации (float).

        """
        y_pred = np.dot(X, self.weights)
        y_mean = np.mean(y)
        return 1 - sum(np.square(y - y_pred)) / sum(np.square(y - y_mean))

    def get_best_score(self) -> Optional[float]:
        """
        Возвращает метрику обученной модели.

        Returns:
            Значение метрики (float) или None, если метрика не вычислена.
        """
        return self.best_score

    def get_coef(self) -> np.ndarray:
        """
        Возвращает коэффициенты модели (без bias).

        Returns:
           np.ndarray: Вектор коэффициентов модели.

        """
        return self.weights[1:]
