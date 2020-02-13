from .linear import Linear
from .poisson_regression import PoissonRegression


def get_function(args):
    if args.function_name == "poisson_regression":
        return PoissonRegression()
    elif args.function_name == "linear":
        return Linear()
    else:
        raise RuntimeError(f"This type of function doesn't support: {args.function_name}")
