import numpy as np
import scipy.optimize

from src.main.optim_methods.interface_method_optim import InterfaceMethodOptim, InterfaceOptimAnswer
from src.main.stop_conditions.common import NumIterStopCondition, InterfaceStopCondition


class BFGS(InterfaceMethodOptim):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        F = self._w.shape[0]
        self._C = np.identity(F)
        self._grad = self._function.loss_gradient(self._w, self._X, self._y)

    def step(self):
        S, F = self._X.shape

        self._w, self._grad, self.C, lr = self._algo(self._w, self._grad, self._C)
        # BAD SMELL, max_bound like magic const
        self._tensorboard_part(lr)

    def get_answer(self):
        return InterfaceOptimAnswer(self._w_start, self._w, self._function)

    # MOCK. may be should be rewritten
    def _get_stop_condition(self, F) -> InterfaceStopCondition:
        return NumIterStopCondition(F)

    def _algo(self, x, grad, C):
        """
        https://ru.wikipedia.org/wiki/%D0%90%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC_%D0%91%D1%80%D0%BE%D0%B9%D0%B4%D0%B5%D0%BD%D0%B0_%E2%80%94_%D0%A4%D0%BB%D0%B5%D1%82%D1%87%D0%B5%D1%80%D0%B0_%E2%80%94_%D0%93%D0%BE%D0%BB%D1%8C%D0%B4%D1%84%D0%B0%D1%80%D0%B1%D0%B0_%E2%80%94_%D0%A8%D0%B0%D0%BD%D0%BD%D0%BE
        :param grad:
        :param hessian:
        :param w_start:
        :return:
        """
        F = x.shape[0]
        assert x.shape == (F, 1)
        assert grad.shape == (F, 1)
        assert C.shape == (F, F)
        I = np.identity(F)
        # BAD SMELL
        max_bound = 1

        # MOCK. formulae.
        p = -C.dot(grad)

        def optim_func(lr: float) -> float:
            return self._function.loss(x + lr * p, self._X, self._y).item()

        # bounds write like BAD SMELL
        res = scipy.optimize.minimize_scalar(optim_func,
                                             method="bounded",
                                             bounds=(-0.5, max_bound)
                                             )
        assert res['success'] is True
        # MOCK. должно удовлетворять условия Вольфе
        lr = res['x']

        x_next = x + lr * p  # (F, 1)
        s = x_next - x  # (F, 1)
        grad_next = self._function.loss_gradient(x_next, self._X, self._y)
        y = grad_next - grad  # (F, 1)

        divider1 = y.T.dot(s).item()
        # BAD SMELL
        assert divider1 != 0
        ro = 1 / divider1  # (1, 1)

        first1 = I - ro * s.dot(y.T)  # (F, F)
        first2 = I - ro * y.dot(s.T)  # (F, F)
        first = first1.dot(C).dot(first2)  # (F, F)

        second = ro * s.dot(s.T)  # (F, 1)
        C_next = first + second
        return x_next, grad_next, C_next, lr
