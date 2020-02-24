import numpy as np
import scipy
from numpy.linalg import LinAlgError

from src.main.optim_methods.interface_method_optim import InterfaceMethodOptim, InterfaceOptimAnswer
from src.main.optim_methods.utils import get_lr


class Newton(InterfaceMethodOptim):
    def step(self):
        S, F = self._X.shape
        grad = self._function.loss_gradient(self._w, self._X, self._y)
        assert grad.shape == (F, 1)
        hessian = self._function.loss_hessian(self._w, self._X, self._y)
        assert hessian.shape == (F, F)
        assert self._w.shape == (F, 1)
        # coef = np.linalg.inv(hessian)
        done = False
        eps = 0
        coef = None

        while not done:
            try:
                coef = self._get_coef(hessian, add_eps_coef=eps)
            except LinAlgError:
                # BAD SMELL
                eps += 1e-6
                continue
            done = True

        assert coef is not None
        assert coef.shape == (F, F)
        direction = coef.dot(grad)
        # BAD SMELL, max_bound like magic const
        lr = get_lr(self._w, direction, self._X, self._y, self._function, max_bound=1000)
        self._w -= lr * direction
        self._tensorboard_part(lr)

    def get_answer(self):
        return InterfaceOptimAnswer(self._w_start, self._w, self._function)

    def _get_coef(self, hessian, add_eps_coef):
        '''
            A * X = B

            L Y = B
            L^(T) X = Y
        '''
        F = hessian.shape[0]
        I = np.identity(F)
        B = I

        L = scipy.linalg.cholesky(hessian + I * add_eps_coef, lower=True)
        Y = scipy.linalg.solve(L, B)
        X = scipy.linalg.solve(L.T, Y)
        return X
