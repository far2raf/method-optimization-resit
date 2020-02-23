import numpy as np
import scipy.optimize

from src.main.optim_methods.interface_method_optim import InterfaceMethodOptim, InterfaceOptimAnswer


class L1Prox(InterfaceMethodOptim):

    def __init__(self, *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

    def step(self):
        S, F = self._X.shape
        pure_grad = self._function.loss_pure_gradient(self._w, self._X, self._y)
        l1 = self._function.get_l1()
        assert pure_grad.shape == (F, 1)
        self._w, lr = self._algo(self._w, pure_grad, l1)

        self._tensorboard_part(lr)

    def get_answer(self):
        return InterfaceOptimAnswer(self._w_start, self._w, self._function)

    def _algo(self, w, grad_F, l1):
        """
        https://en.wikipedia.org/wiki/Proximal_gradient_methods_for_learning#Fixed_point_iterative_schemes
        """

        def prox_gamma_l1(gamma, x):
            first = np.where(x > l1 * gamma, x - l1 * gamma , 0)
            second = np.where(x < l1 * gamma, x + l1 * gamma, 0)
            return first + second

        function = self._function
        X = self._X
        y = self._y

        def optim_func(lr: float) -> float:
            gamma = lr
            new_w = prox_gamma_l1(gamma, w - gamma * grad_F)
            return function.loss(new_w, X, y).item()

        # bounds write like BAD SMELL
        max_bound = 1
        res = scipy.optimize.minimize_scalar(optim_func,
                                             method="bounded",
                                             bounds=(-0.5, max_bound)
                                             )
        assert res['success'] is True
        lr = res['x']
        gamma = lr
        new_w = prox_gamma_l1(gamma, w - gamma * grad_F)

        return new_w, lr
