import functools
import operator

import tqdm as tqdm


def get_eps_condition(args):
    if args.type_of_eps_stop_condition == "none":
        return None
    else:
        raise RuntimeError(f"This eps condition undefined: {args.type_of_eps_condition}")


def get_stop_condition(args):
    if args.no_use_num_stop_condition and args.type_of_eps_condition == "none":
        raise RuntimeError("no condition")
    iter_condition = None if args.no_use_num_stop_condition else \
        NumIterStopCondition(size=args.max_num_iter, use_tqdm=not args.no_use_tqdm)
    eps_condition = get_eps_condition(args)
    if eps_condition and iter_condition:
        return OrStopCondition((eps_condition, iter_condition))
    else:
        return eps_condition if eps_condition else iter_condition


class InterfaceStopCondition:

    def check(self, w) -> bool:
        raise RuntimeError("Method should be overridden")


class OrStopCondition(InterfaceStopCondition):

    def __init__(self, conditions):
        self._conditions = conditions

    def check(self, w):
        results = map(lambda condition: condition(w), self._conditions)
        return functools.reduce(operator.or_, results)


class NumIterStopCondition(InterfaceStopCondition):

    def __init__(self, size, use_tqdm=False):
        self._use_tqdm = use_tqdm
        self._tqdm = tqdm.tqdm(total=size, position=0)
        self._size = size
        self._counter = 0

    def check(self, w):
        if self._counter < self._size:
            if self._use_tqdm:
                self._tqdm.update(1)
            self._counter += 1
            return False
        else:
            return True


class EpsBetweenParamStopCondition(InterfaceStopCondition):

    def __init__(self, eps):
        self._eps = eps
        self._previous_w = None

    def check(self, w) -> bool:
        result = False
        if self._previous_w is not None:
            diff = ((w - self._previous_w) ** 2).sum()
            if diff < self._eps:
                result = True
        self._previous_w = w
        return result
