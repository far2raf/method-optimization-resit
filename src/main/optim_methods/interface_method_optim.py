# MOCK


class InterfaceOptimAnswer:

    # MOCK
    def generate_json(self):
        pass

    def get_initial_point(self):
        raise RuntimeError("Method should be overridden")

    def get_optimal_point(self):
        raise RuntimeError("Method should be overridden")

    def get_func_value(self):
        raise RuntimeError("Method should be overridden")

    def get_gradient_value(self):
        raise RuntimeError("Method should be overridden")

    def get_stopping_criterian(self):
        raise RuntimeError("Method should be overridden")

    def get_num_nonzero_components(self):
        if self.is_sparse_answer():
            raise RuntimeError(f"Method can't be invoke with not sparse answer")
        else:
            raise RuntimeError("Method should be overridden")

    def is_sparse_answer(self):
        raise RuntimeError("Method should be overridden")


class InterfaceMethodOptim:

    def run(self, X, y, function) -> InterfaceOptimAnswer:
        raise RuntimeError("Method should be overridden")
