from src.main.optim_methods.adam import Adam
from src.main.optim_methods.bfgs import BFGS
from src.main.optim_methods.gradientdescent import GradientDescent
from src.main.optim_methods.hessian_free_newton import HessianFreeNewton
from src.main.optim_methods.l1prox import L1Prox
from src.main.optim_methods.lbfgs import LBFGS
from src.main.optim_methods.newton import Newton


def get_opt_method_maker(program_running_arguments, tensorboard_writer):
    name = program_running_arguments.optim_method
    common_kwargs = {
        "tensorboard_writer": tensorboard_writer,
        "eps_for_zero_division": program_running_arguments.eps_for_zero_division
    }
    if name == "gradient":
        return lambda *args: GradientDescent(*args, **common_kwargs)
    elif name == "newton":
        return lambda *args: Newton(*args, **common_kwargs)
    elif name == "hfn":
        return lambda *args: HessianFreeNewton(*args, **common_kwargs)
    elif name == "bfgs":
        return lambda *args: BFGS(*args, **common_kwargs)
    elif name == "lbfgs":
        return lambda *args: LBFGS(*args,
                                   size=program_running_arguments.lbfgs_history_size,
                                   **common_kwargs
                                   )
    elif name == "adam":
        return lambda *args: Adam(*args,
                                  betta1=program_running_arguments.betta1,
                                  betta2=program_running_arguments.betta2,
                                  lr=program_running_arguments.lr,
                                  **common_kwargs
                                  )
    elif name == "l1prox":
        return lambda *args: L1Prox(*args,
                                    lr=program_running_arguments.lr,
                                    **common_kwargs)
    else:
        raise RuntimeError(f"Optim method {name} undefined")
