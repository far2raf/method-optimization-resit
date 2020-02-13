

class BaseFunction:

    def function(self):
        raise RuntimeError("Should be overridden")

    def gradient(self):
        raise RuntimeError("Should be overridden")

    def guassian(self):
        raise RuntimeError("Should be overridden")
