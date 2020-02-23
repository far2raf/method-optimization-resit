import numpy as np

from src.main.optim_methods.interface_method_optim import InterfaceMethodOptim, InterfaceOptimAnswer
from src.main.stop_conditions.common import NumIterStopCondition, InterfaceStopCondition


class Adam(InterfaceMethodOptim):

    def __init__(self, *args,
                 beta1: float,
                 beta2: float,
                 lr: float,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        F = self._w.shape[0]
        self._beta1 = beta1
        self._beta2 = beta2
        self._mw = np.zeros((F, 1))
        self._uw = np.zeros((F, 1))
        self._lr = lr

    def step(self):
        S, F = self._X.shape
        grad = self._function.loss_gradient(self._w, self._X, self._y)
        assert grad.shape == (F, 1)
        self._w = self._algo(self._w, grad)

        # For RELEASE version should be deleted
        loss = self._function.loss(self._w, self._X, self._y)
        self._tensorboard_writer.add_scalar('loss', loss, self._learning_step)

    def get_answer(self):
        return InterfaceOptimAnswer(self._w_start, self._w, self._function)

    # MOCK. may be should be rewritten
    def _get_stop_condition(self, F) -> InterfaceStopCondition:
        return NumIterStopCondition(F)

    def _algo(self, w, grad):
        """
        https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam
        """
        beta1 = self._beta1
        beta2 = self._beta2
        eta = self._lr
        mw = self._mw
        uw = self._uw

        new_mw = beta1 * mw + (1 - beta1) * grad
        new_uw = beta2 * uw + (1 - beta2) * grad ** 2

        _mw = new_mw / (1 - beta1)
        _uw = new_uw / (1 - beta2)

        new_w = w - eta * _mw / np.sqrt(_uw + self._eps_for_zero_division)

        return new_w
