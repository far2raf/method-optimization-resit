from src.main.optim_methods.sgd import SGD


def get_method(args):
    name = args.optimize_method
    if name == "sgd":
        return SGD()
    else:
        raise RuntimeError(f"Optim method {name} undefined")
