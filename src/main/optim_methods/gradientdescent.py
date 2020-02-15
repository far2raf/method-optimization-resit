from src.main.functions.interface_function import InterfaceFunction
from src.main.optim_methods.interface_method_optim import InterfaceMethodOptim, InterfaceOptimAnswer
from src.main.stop_conditions.common import InterfaceStopCondition


class GradientDescent(InterfaceMethodOptim):

    def __init__(self, X, y, function: InterfaceFunction, stop_condition: InterfaceStopCondition, lr):
        super().__init__(X, y, function, stop_condition)
        self._lr = lr

    def step(self):
        grad = self._function.loss_gradient(self._w, self._X, self._y)
        assert grad.shape == self._w.shape
        self._w -= self._lr * grad

    def get_answer(self):
        return InterfaceOptimAnswer(self._w_start, self._w, self._function)

