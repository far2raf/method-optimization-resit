import numpy as np


class BaseFunction:

    def function(self):
        raise RuntimeError("Should be overridden")

    def gradient(self):
        raise RuntimeError("Should be overridden")

    def guassian(self):
        raise RuntimeError("Should be overridden")

    def _regularization_part(self, w, l1, l2):
        l1_part = np.abs(w).mean()
        l2_part = (w ** 2).mean()
        return l1 * l1_part + l2 / 2 * l2_part
