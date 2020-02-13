# MOCK
from src.main.function.interface_function import InterfaceFunction


class Linear(InterfaceFunction):

    # MOCK. Should be checked all shapes
    def _loss(self, X, y, w, l1, l2):
        diff = y - X.dot(w)  # Size (num_of_samples, 1)
        main = (diff ** 2).mean()
        total = main + self._regularization_part(w, l1, l2)
        return total
