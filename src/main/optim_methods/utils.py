import scipy.optimize


def get_lr(w, direction, X, y, function, *args, max_bound):
    # MOCK, thought: what will be better when direction zero (or like eps error). May be just zero.
    # MOCK. BUG: here sometimes lr becomes ndarray, not float
    def optim_func(lr: float) -> float:
        return function.loss(w - lr * direction, X, y).item()

    # bounds write like BAD SMELL
    res = scipy.optimize.minimize_scalar(optim_func,
                                         method="bounded",
                                         bounds=(-0.5, max_bound)
                                         )
    assert res['success'] is True
    lr = res['x']
    return lr
