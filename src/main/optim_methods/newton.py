import numpy as np

from src.main.optim_methods.interface_method_optim import InterfaceMethodOptim, InterfaceOptimAnswer


class Newton(InterfaceMethodOptim):
    def step(self):
        S, F = self._X.shape
        grad = self._function.loss_gradient(self._w, self._X, self._y)
        assert grad.shape == (F, 1)
        hessian = self._function.loss_hessian(self._w, self._X, self._y)
        assert hessian.shape == (F, F)
        assert self._w.shape == (F, 1)
        # в реализации нужно учесть ситуации с плохой обусловленностью
        # и уметь обрабатывать наличие неположительных собственных чисел
        # в гессиане) MOCK. NOT DONE
        coef = np.linalg.inv(hessian)
        self._w -= coef.dot(grad)

    def get_answer(self):
        return InterfaceOptimAnswer(self._w_start, self._w, self._function)
