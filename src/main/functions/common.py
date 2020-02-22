from .linear import Linear
from .poisson_regression import PoissonRegression


def get_function(args):
    common_kwargs = {
        "l1": args.l1,
        "l2": args.l2
    }
    if args.function_name == "poisson_regression":
        return PoissonRegression(**common_kwargs)
    elif args.function_name == "linear":
        return Linear(**common_kwargs)
    else:
        raise RuntimeError(f"This type of function doesn't support: {args.function_name}")
