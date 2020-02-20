import scipy
import scipy.optimize

from src.main.functions.interface_function import InterfaceFunction
from src.main.optim_methods.interface_method_optim import InterfaceMethodOptim, InterfaceOptimAnswer
from src.main.stop_conditions.common import InterfaceStopCondition
import numpy as np


class GradientDescent(InterfaceMethodOptim):

    def __init__(self, X, y, function: InterfaceFunction, stop_condition: InterfaceStopCondition, *args, **kwargs):
        super().__init__(X, y, function, stop_condition, *args, **kwargs)

    def step(self):
        grad = self._function.loss_gradient(self._w, self._X, self._y)
        assert grad.shape == self._w.shape
        optim_func = lambda dw: self._function.loss(self._w - dw * grad, self._X, self._y)
        # May be there bounds write like BAD SMELL
        res = scipy.optimize.minimize_scalar(optim_func, method="bounded", bounds=(0, 1))
        lr = res['x']
        self._w -= lr * grad
        self._tensorboard_part(lr)


    def get_answer(self):
        return InterfaceOptimAnswer(self._w_start, self._w, self._function)

