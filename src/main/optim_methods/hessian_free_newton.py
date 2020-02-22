import numpy as np

from src.main.optim_methods.interface_method_optim import InterfaceMethodOptim, InterfaceOptimAnswer
from src.main.optim_methods.utils import get_lr
from src.main.stop_conditions.common import InterfaceStopCondition, NumIterStopCondition


class HessianFreeNewton(InterfaceMethodOptim):

    def step(self):
        S, F = self._X.shape
        grad = self._function.loss_gradient(self._w, self._X, self._y)
        assert grad.shape == (F, 1)
        hessian = self._function.loss_hessian(self._w, self._X, self._y)
        assert hessian.shape == (F, F)
        assert self._w.shape == (F, 1)

        direction = self._direction(grad, hessian, self._w)

        # BAD SMELL, max_bound like magic const
        lr = get_lr(self._w, direction, self._X, self._y, self._function, max_bound=1000)
        self._w -= lr * direction

        self._tensorboard_part(lr)

    def get_answer(self):
        return InterfaceOptimAnswer(self._w_start, self._w, self._function)

    # MOCK. may be should be rewritten
    def _get_stop_condition(self, F) -> InterfaceStopCondition:
        return NumIterStopCondition(F)

    def _direction(self, grad, hessian, w_start):
        """
        solve Ax = b
        https://habr.com/ru/post/350794/

        :param grad:
        :param hessian:
        :param w_start:
        :return:
        """
        F = grad.shape[0]
        assert grad.shape == (F, 1)
        assert hessian.shape == (F, F)
        assert w_start.shape == (F, 1)
        A = hessian.copy()
        x = np.random.rand(F, 1)
        b = grad.copy()
        r = b - A.dot(x)  # residual
        d = r  # search direction
        stop_condition = self._get_stop_condition(F=F)
        while not stop_condition.finish(x):
            divider1 = d.T.dot(A).dot(d)
            assert divider1.item() != 0  # MOCK. strange check, better with some eps
            alpha = r.T.dot(r) / divider1  # (1, 1)
            x_next = x + d.dot(alpha)
            r_next = r - A.dot(d).dot(alpha)
            divider2 = r.T.dot(r)
            assert divider2.item() != 0  # MOCK. strange check, better with some eps
            betta = r_next.T.dot(r_next) / divider2
            d_next = r_next + d.dot(betta)

            x = x_next
            r = r_next
            d = d_next

        return x
