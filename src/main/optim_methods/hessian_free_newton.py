import numpy as np

from src.main.optim_methods.interface_method_optim import InterfaceMethodOptim, InterfaceOptimAnswer
from src.main.optim_methods.utils import get_lr
from src.main.stop_conditions.common import InterfaceStopCondition, NumIterStopCondition


class HessianFreeNewton(InterfaceMethodOptim):

    def step(self):
        S, F = self._X.shape
        grad = self._function.loss_gradient(self._w, self._X, self._y)
        assert grad.shape == (F, 1)
        assert self._w.shape == (F, 1)

        direction = self._direction(grad, self._w)

        # BAD SMELL, max_bound like magic const
        max_bound = 1
        lr = get_lr(self._w, direction, self._X, self._y, self._function, max_bound=max_bound)
        self._w -= lr * direction

        self._tensorboard_part(lr)

    def get_answer(self):
        return InterfaceOptimAnswer(self._w_start, self._w, self._function)

    # MOCK. probably should be rewritten
    def _get_stop_condition(self, F) -> InterfaceStopCondition:
        return NumIterStopCondition(F)

    def _direction(self, grad, w_start):
        """
        https://habr.com/ru/post/350794/
        http://andrew.gibiansky.com/blog/machine-learning/hessian-free-optimization/
        """
        F = grad.shape[0]
        assert grad.shape == (F, 1)
        assert w_start.shape == (F, 1)

        def A_dot_vec(x):
            # BAD SMELL
            eps = 1e-6
            new_w = w_start + eps * x
            new_grad = self._function.loss_gradient(new_w, self._X, self._y)
            res = (new_grad - grad) / eps
            assert not np.isnan(res).any()
            return res

        x = np.random.rand(F, 1)
        b = grad.copy()
        r = b - A_dot_vec(x)  # residual
        d = r  # search direction
        stop_condition = self._get_stop_condition(F=F)
        while not stop_condition.finish(x):

            divider1 = d.T.dot(A_dot_vec(d))
            alpha = r.T.dot(r) / (divider1 + self._eps_for_zero_division) # (1, 1)

            x_next = x + d.dot(alpha)
            r_next = r - A_dot_vec(d).dot(alpha)

            divider2 = r.T.dot(r)
            beta = r_next.T.dot(r_next) / (divider2 + self._eps_for_zero_division)

            d_next = r_next + d.dot(beta)

            x = x_next
            r = r_next
            d = d_next

        w_next = x

        return w_next
