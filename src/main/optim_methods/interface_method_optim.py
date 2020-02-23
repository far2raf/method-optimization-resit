import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.main.functions.interface_function import InterfaceFunction
from src.main.stop_conditions.common import InterfaceStopCondition


class InterfaceOptimAnswer:

    def __init__(self, initial_point, optimal_point, func: InterfaceFunction) -> None:
        super().__init__()
        self._func = func
        self._optimal_point = optimal_point
        self._initial_point = initial_point

    def generate_json(self):
        raise RuntimeError("MOCK")

    def get_initial_point(self):
        return self._initial_point

    def get_optimal_point(self):
        return self._optimal_point

    def get_loss_value(self):
        raise RuntimeError("Method should be overridden")

    def get_gradient_loss_value(self):
        raise RuntimeError("Method should be overridden")

    def get_stopping_criterion(self):
        raise RuntimeError("Method should be overridden")

    def get_num_nonzero_components(self):
        if self.is_sparse_answer():
            raise RuntimeError(f"Method can't be invoke with not sparse answer")
        else:
            raise RuntimeError("Method should be overridden")

    def is_sparse_answer(self):
        raise RuntimeError("Method should be overridden")


class InterfaceMethodOptim:

    def __init__(self,
                 X,
                 y,
                 function: InterfaceFunction,
                 stop_condition: InterfaceStopCondition,
                 tensorboard_writer: SummaryWriter,
                 eps_for_zero_division: float
                 ):
        self._eps_for_zero_division = eps_for_zero_division
        assert len(X.shape) == 2
        num_of_samples, num_of_features = X.shape
        assert y.shape == (num_of_samples, 1)
        self._X = X.copy()
        self._y = y.copy()
        self._function = function
        self._stop_condition = stop_condition
        self._w_start = np.random.rand(num_of_features, 1)
        self._w = self._w_start.copy()

        # For tensorboard
        self._tensorboard_writer = tensorboard_writer
        self._learning_step = 0

    def run(self) -> InterfaceOptimAnswer:
        while not self._stop_condition.finish(self._w):
            self.step()
            self._learning_step += 1
        return self.get_answer()

    def step(self):
        raise RuntimeError("Method should be overridden")

    def get_answer(self):
        raise RuntimeError("Method should be overridden")

    def _name(self):
        raise RuntimeError("Method should be overridden")

    def _tensorboard_part(self, lr):
        # For RELEASE version should be deleted
        self._tensorboard_writer.add_scalar('lr', lr, self._learning_step)
        loss = self._function.loss(self._w, self._X, self._y)
        self._tensorboard_writer.add_scalar('loss', loss, self._learning_step)
