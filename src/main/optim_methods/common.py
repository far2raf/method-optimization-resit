from src.main.optim_methods.gradientdescent import GradientDescent
from src.main.optim_methods.hessian_free_newton import HessianFreeNewton
from src.main.optim_methods.newton import Newton


def get_opt_method_maker(args):
    name = args.optim_method
    if name == "gradient":
        return lambda *grad_args: GradientDescent(*grad_args, lr=args.lr)
    elif name == "newton":
        return Newton
    elif name == "hfn":
        return HessianFreeNewton
    else:
        raise RuntimeError(f"Optim method {name} undefined")
