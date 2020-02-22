from src.main.optim_methods.bfgs import BFGS
from src.main.optim_methods.gradientdescent import GradientDescent
from src.main.optim_methods.hessian_free_newton import HessianFreeNewton
from src.main.optim_methods.lbfgs import LBFGS
from src.main.optim_methods.newton import Newton


def get_opt_method_maker(program_running_arguments, tensorboard_writer):
    name = program_running_arguments.optim_method
    if name == "gradient":
        return lambda *args: GradientDescent(*args, tensorboard_writer=tensorboard_writer)
    elif name == "newton":
        return lambda *args: Newton(*args, tensorboard_writer=tensorboard_writer)
    elif name == "hfn":
        return lambda *args: HessianFreeNewton(*args, tensorboard_writer=tensorboard_writer)
    elif name == "bfgs":
        return lambda *args: BFGS(*args, tensorboard_writer=tensorboard_writer)
    elif name == "lbfgs":
        return lambda *args: LBFGS(*args,
                                   tensorboard_writer=tensorboard_writer,
                                   size=program_running_arguments.lbfgs_history_size
                                   )
    elif name == "adam":
        raise RuntimeError("MOCK, not realized yet")
    elif name == "l1prox":
        raise RuntimeError("MOCK, not realized yet")
    else:
        raise RuntimeError(f"Optim method {name} undefined")
