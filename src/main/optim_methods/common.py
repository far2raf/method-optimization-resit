from src.main.optim_methods.gradientdescent import GradientDescent


def get_opt_method_maker(args):
    name = args.optim_method
    if name == "gradient":
        return lambda *grad_args: GradientDescent(*grad_args, lr=args.lr)
    else:
        raise RuntimeError(f"Optim method {name} undefined")
