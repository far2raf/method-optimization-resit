# MOCK
import numpy as np

from src.main.functions.interface_function import InterfaceFunction


class PoissonRegression(InterfaceFunction):
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
        total = main + self._loss_regularization_part(w)
        return total

    def _loss_gradient(self, w, X, y):
        S, F = X.shape
        xw = X.dot(w)
        exp_part = np.exp(xw)
        diff = y - exp_part
        main = 1 / S * X.T.dot(diff)
        total = main + self._loss_gradient_regularization_part(w)
        return total

    def _loss_hessian(self, w, X, y):
        xw = X.dot(w)
        exp_part = np.exp(xw)
        # MOCK should be checked
        M = np.diag(exp_part) # (S, S)
        main = X.t.dot(M).dot(X)
        total = main + self._loss_hessian_regularization_part(w)
        return total
