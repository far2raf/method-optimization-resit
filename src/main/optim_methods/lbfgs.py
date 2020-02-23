import numpy as np

from src.main.optim_methods.interface_method_optim import InterfaceMethodOptim, InterfaceOptimAnswer
from src.main.stop_conditions.common import NumIterStopCondition, InterfaceStopCondition


class LBFGS(InterfaceMethodOptim):

    def __init__(self, *args,
                 size: int,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self._size = size
        F = self._w.shape[0]
        self._buffer_H = []
        self._buffer_s = []
        self._buffer_y = []
        self._buffer_ro = []
        self._buffer_grad = [self._function.loss_gradient(self._w, self._X, self._y)]

    def step(self):
        S, F = self._X.shape

        new_w, new_grad, new_H0, new_s, new_y, new_ro = self._algo()
        self._w = new_w
        self._buffer_grad.append(new_grad)
        self._buffer_H.append(new_H0)
        self._buffer_s.append(new_s)
        self._buffer_y.append(new_y)
        self._buffer_ro.append(new_ro)

        loss = self._function.loss(self._w, self._X, self._y)
        self._tensorboard_writer.add_scalar('loss', loss, self._learning_step)

    def get_answer(self):
        return InterfaceOptimAnswer(self._w_start, self._w, self._function)

    # MOCK. may be should be rewritten
    def _get_stop_condition(self, F) -> InterfaceStopCondition:
        return NumIterStopCondition(F)

    def _algo(self):
        """
        https://en.wikipedia.org/wiki/Limited-memory_BFGS
        """
        F = self._w.shape[0]
        I = np.identity(F)
        k = self._learning_step
        m = min(self._size, k)

        alpha = [0] * k
        beta = [0] * k
        H0 = self._buffer_H
        s = self._buffer_s
        y = self._buffer_y
        ro = self._buffer_ro
        g = self._buffer_grad
        q = g[k]

        for i in range(k - 1, k - m - 1, -1):
            alpha[i] = ro[i] * s[i].T.dot(q).item()
            q = q - alpha[i] * y[i]

        if k != 0:
            divider1 = y[k - 1].T.dot(y[k - 1]).item()
            gamma = s[k - 1].T.dot(y[k - 1]).item() / (divider1 + self._eps_for_zero_division)
        else:
            gamma = 1

        new_H0 = gamma * I
        z = new_H0.dot(q)
        for i in range(k - m, k, 1):
            beta[i] = ro[i] * y[i].T.dot(z).item()
            z = z + s[i] * (alpha[i] - beta[i])

        new_z = z
        new_w = self._w - new_z
        new_s = new_w - self._w
        new_grad = self._function.loss_gradient(new_w, self._X, self._y)
        new_y = new_grad - g[-1]
        divider2 = new_y.T.dot(new_s)
        new_ro = 1 / (divider2 + self._eps_for_zero_division)

        return new_w, new_grad, new_H0, new_s, new_y, new_ro
