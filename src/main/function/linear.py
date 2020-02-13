# MOCK
from src.main.function.base_function import BaseFunction


class Linear(BaseFunction):

    # MOCK. Should be checked all shapes
    def _loss(self, X, y, w, l1, l2):
        diff = y - X.dot(w)  # Size (num_of_samples, 1)
        main = (diff ** 2).mean()
        total = main + self._regularization_part(w, l1, l2)
        return total
