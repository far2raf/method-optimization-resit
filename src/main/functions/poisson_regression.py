# MOCK
import numpy as np

from src.main.functions.interface_function import InterfaceFunction


class PoissonRegression(InterfaceFunction):
    pass

    # MOCK. should check all shapes
    # def _loss(self, w, X, y, l1, l2):
    #     """
    #
    #    :param X: Size: (num_samples, num_features)
    #    :param y: Size: (num_samples, 1)
    #    :param w: Size: (num_features, 1)
    #    :param l1: float
    #    :param l2: float
    #    :return:
    #    """
    #     mean_y = X.dot(w)  # (num_samples, 1)
    #     second = np.exp(mean_y)  # (num_samples, 1)
    #     first = y * mean_y  # (num_samples, 1)
    #     main = (second - first).mean()
    #
    #     total = main + self._loss_regularization_part(w, l1, l2)
    #     return total
