import numpy as np
import scipy.sparse

from src.main.functions.interface_function import InterfaceFunction


class PoissonRegression(InterfaceFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # В формулах в задании и на вики почему-то максимизирует loss
        # Для наглядности сделан специальный параметр меняющий выражения для минимизации
        self._maximization_to_minimization = -1

    def _function(self, w, X):
        mean = np.exp(X.dot(w))
        return np.random.poisson(mean)

    def _loss(self, w, X, y):
        S, F = X.shape
        xw = X.dot(w)
        first = y.T.dot(xw)
        exp_part = np.exp(xw)
        second = np.ones((S, 1)).T.dot(exp_part)
        main = (first - second) / S
        total = self._maximization_to_minimization * main + self._loss_regularization_part(w)
        return total

    def _loss_pure_gradient(self, w, X, y):
        S, F = X.shape
        xw = X.dot(w)
        exp_part = np.exp(xw)
        diff = y - exp_part
        main = 1 / S * X.T.dot(diff)
        return self._maximization_to_minimization * main

    def _loss_hessian(self, w, X, y):
        F = w.shape[0]

        xw = X.dot(w)
        exp_part = np.exp(xw)
        # MOCK should be checked
        M = scipy.sparse.diags([exp_part.view().reshape(-1)], [0])  # (S, S)
        main = -X.T.dot(M).dot(X)

        assert main.shape == (F, F)

        total = self._maximization_to_minimization * main + self._loss_hessian_regularization_part(w)
        return total
