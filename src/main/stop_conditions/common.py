import functools
import operator

import tqdm as tqdm


def get_eps_condition(args, F):
    if args.type_of_eps_stop_condition == "none":
        return None
    elif args.type_of_eps_stop_condition == "eps_between_param":
        return EpsBetweenParamStopCondition(args.eps_stop_condition, F)
    else:
        raise RuntimeError(f"This eps condition undefined: {args.type_of_eps_condition}")


def get_stop_condition(args, F):
    if args.no_use_num_stop_condition and args.type_of_eps_condition == "none":
        raise RuntimeError("no condition")
    iter_condition = None if args.no_use_num_stop_condition else \
        NumIterStopCondition(size=args.max_num_iter, use_tqdm=not args.no_use_tqdm)
    eps_condition = get_eps_condition(args, F)
    if eps_condition and iter_condition:
        return OrStopCondition((eps_condition, iter_condition))
    else:
        return eps_condition if eps_condition else iter_condition


class InterfaceStopCondition:

    def finish(self, w) -> bool:
        raise RuntimeError("Method should be overridden")


class OrStopCondition(InterfaceStopCondition):

    def __init__(self, conditions):
        self._conditions = conditions

    def finish(self, w):
        results = map(lambda condition: condition.finish(w), self._conditions)
        return functools.reduce(operator.or_, results)


class NumIterStopCondition(InterfaceStopCondition):

    def __init__(self, size, use_tqdm=False):
        self._use_tqdm = use_tqdm
        if use_tqdm:
            self._tqdm = tqdm.tqdm(total=size, position=0)
        self._size = size
        self._counter = 0

    def finish(self, w):
        if self._counter < self._size:
            if self._use_tqdm:
                self._tqdm.update(1)
            self._counter += 1
            return False
        else:
            return True


class EpsBetweenParamStopCondition(InterfaceStopCondition):

    def __init__(self, eps, F):
        self._F = F
        self._eps = eps
        self._previous_w = None

    def finish(self, w) -> bool:
        assert w.shape == (self._F, 1)
        result = False
        if self._previous_w is not None:
            diff = (w - self._previous_w)
            mean = diff.T.dot(diff) / self._F
            if mean < self._eps:
                result = True
        self._previous_w = w
        return result
