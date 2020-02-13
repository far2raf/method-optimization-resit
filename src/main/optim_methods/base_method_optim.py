# MOCK


class BaseOptimAnswer:
    pass


class BaseMethodOptim:

    def run(self, X, y, function) -> BaseOptimAnswer:
        raise RuntimeError("Method run shold be overridden")
