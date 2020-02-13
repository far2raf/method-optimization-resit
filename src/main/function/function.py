from .poisson_regression import PoissonRegression


def get_function(args):
    if args.function_name == "poisson_regression":
        return PoissonRegression()
    else:
        raise RuntimeError(f"This type of function doesn't support: {name}")
