from src.main.optim_methods.interface_method_optim import InterfaceMethodOptim, InterfaceOptimAnswer
from src.main.stop_conditions.common import InterfaceStopCondition, NumIterStopCondition, EpsBetweenParamStopCondition, \
    OrStopCondition


class HessianFreeNewton(InterfaceMethodOptim):
    def step(self):
        S, F = self._X.shape
        grad = self._function.loss_gradient(self._w, self._X, self._y)
        assert grad.shape == (F, 1)
        hessian = self._function.loss_hessian(self._w, self._X, self._y)
        assert hessian.shape == (F, F)
        assert self._w.shape == (F, 1)
        self._w -= self._cg(grad, hessian, self._w)

    def get_answer(self):
        return InterfaceOptimAnswer(self._w_start, self._w, self._function)

    def _get_stop_condition(self, F, eps=0.001) -> InterfaceStopCondition:
        eps_condition = EpsBetweenParamStopCondition(eps, F)
        iter_condition = NumIterStopCondition(F)
        return OrStopCondition((eps_condition, iter_condition))

    def _cg(self, grad, hessian, w_start):
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
        x = w_start.copy()
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
