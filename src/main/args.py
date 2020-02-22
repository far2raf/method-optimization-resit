import argparse
import json
import os


def load_args_settings(args):
    argparse_dict = vars(args)
    path = os.path.abspath(f"{args.args_settings_folder}/{args.function_name}-{args.optim_method}.json")
    _json = json.load(open(path, "r"))
    argparse_dict.update(_json)


argument_parser = argparse.ArgumentParser()

# args settings
argument_parser.add_argument("--no_use_save_args_settings", action="store_true")
argument_parser.add_argument("--args_settings_folder", type=str, default="src/main/args_settings")

# common
argument_parser.add_argument("--data_folder", type=str, default="data")
argument_parser.add_argument("--seed", type=int, default=42)
argument_parser.add_argument("--function_name", type=str,
                             default="linear"
                             # default="poisson_regression"
                             )

optim_methods = {'gradient', 'newton', 'hfn', 'bfgs', 'lbfgs', 'l1prox', 'adam'}
argument_parser.add_argument("--optim_method",
                             type=str,
                             help=f"high-level optimization method, will be one of {optim_methods}.",
                             choices=optim_methods,
                             default="gradient"  # MOCK. should be deleted
                             # default="newton"
                             # default="hfn"
                             # default="bfgs"
                             # default="lbfgs"
                             # default="adam"
                             # default="l1prox"
                             )

# line_search_method = {'golden_search', 'brent', 'armijo', 'wolfe', 'lipschitz'}
line_search_methods = {"MOCK"}  # MOCK. maybe should be only brent and set it like default
argument_parser.add_argument("--line_search",
                             type=str,
                             help=f"linear optimization method, will be one of {line_search_methods}" +
                                  f"Note that you don't have to support a combination of 'newton' nor 'hfn' nor any 'bfgs' optimization method and 'lipschitz' linear search.")

# eps stop condition
eps_stop_condition_methods = {"none", "eps_between_param"}
argument_parser.add_argument("--type_of_eps_stop_condition", choices=eps_stop_condition_methods, type=str,
                             default="none")
argument_parser.add_argument("--eps_stop_condition", type=float, default=0.00001)

# iter stop condition
argument_parser.add_argument("--no_use_num_stop_condition", action="store_true")
argument_parser.add_argument("--max_num_iter", type=int)
argument_parser.add_argument("--no_use_tqdm", action="store_true")

# only for lbfgs
argument_parser.add_argument("--lbfgs_history_size", type=int, default=10,
                             help="optional for L-BFGS method; number of entries in history.")

# only for adam
argument_parser.add_argument("--betta1", type=float, default=0.9)
argument_parser.add_argument("--betta2", type=float, default=0.999)
argument_parser.add_argument("--lr", type=float, default=0.0001)

argument_parser.add_argument("--eps_for_zero_division", type=float, default=1e-8)

# MOCK
# --point_distribution: string; initial weights distribution class, will be one of {'uniform', 'gaussian'}. In case of uniform its parameters must be (-1, 1) and in case of gaussian its parameters must be (0, $\sqrt{10}$).
# --seed: int; seed for numpy randomness.
# --eps: float; epsilon to use in termination condition.
# --cg-tolerance-policy: string; optional key for HFN method; conjugate gradients method tolerance choice policy, will be one of {'const', 'sqrtGradNorm', 'gradNorm'}.
# --cg-tolerance-eta: float; optional key for HFN method; conjugate gradients method tolerance parameter eta.
# --lbfgs-history-size: int; optional for L-BFGS method; number of entries in history.
# --l1_lambda: float; optional key for l1-proximal method; l1-regularization coefficient.
# --batch_size: int; optional key for SGD method.
# --sgd_n_iters: int, optional key for SGD method.
