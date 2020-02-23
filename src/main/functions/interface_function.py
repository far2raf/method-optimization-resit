import numpy as np


class InterfaceFunction:
    """
        F - number_of_features
        S - number_of_samples
        w - F:1
        X - S:F
        y - S:1
    """

    def __init__(self, l1, l2):
        self._l1 = l1
        self._l2 = l2

    def get_l1(self):
        return self._l2

    def get_l2(self):
        return self._l2

    def function(self, w, X):
        assert len(X.shape) != 2
        S, F = X.shape
        assert w.shape == (F, 1)
        return self._function(w, X)

    def loss(self, w, X, y):
        self._check_all(w, X, y)
        loss = self._loss(w, X, y)
        assert loss.shape == (1, 1)
        return loss

    def loss_pure_gradient(self, w, X, y):
        self._check_all(w, X, y)
        S, F = X.shape
        pure_grad = self._loss_pure_gradient(w, X, y)
        assert pure_grad.shape == (F, 1)
        return pure_grad

    def loss_gradient(self, w, X, y):
        self._check_all(w, X, y)
        S, F = X.shape
        grad = self._loss_gradient(w, X, y)
        assert grad.shape == (F, 1)
        return grad

    def loss_hessian(self, w, X, y):
        self._check_all(w, X, y)
        S, F = X.shape
        hessian = self._loss_hessian(w, X, y)
        assert hessian.shape == (F, F)
        return hessian

    @staticmethod
    def _check_all(w, X, y):
        assert len(X.shape) == 2
        S, F = X.shape
        assert w.shape == (F, 1)
        assert y.shape == (S, 1)

    def _function(self, w, X):
        raise RuntimeError("Should be overridden")

    def _loss(self, w, X, y):
        raise RuntimeError("Should be overridden")

    def _loss_gradient(self, w, X, y):
        first = self._loss_pure_gradient(w, X, y)
        second = self._loss_gradient_regularization_part(w)
        return first + second

    def _loss_pure_gradient(self, w, X, y):
        raise RuntimeError("Should be overridden")

    def _loss_hessian(self, w, X, y):
        raise RuntimeError("Should be overridden")

    def _loss_regularization_part(self, w):
        shape = w.shape[0]
        l1_part = np.abs(w).sum()
        l2_part = w.T.dot(w)
        return self._l1 * l1_part + self._l2 / 2 * l2_part

    def _loss_gradient_regularization_part(self, w):
        shape = w.shape[0]
        # MOCK. better way with l1 is written in telegram
        l1_part = np.sign(w)
        l2_part = 2 * w
        return self._l1 * l1_part + self._l2 / 2 * l2_part

    def _loss_hessian_regularization_part(self, w):
        shape = w.shape[0]
        l2_part = 2 * np.ones((shape, shape))
        return self._l2 / 2 * l2_part
