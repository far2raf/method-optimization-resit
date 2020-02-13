# MOCK, generate_data.py has been changed. Should be recheck
import argparse

argument_parser = argparse.ArgumentParser()

argument_parser.add_argument("--ds_path", type=str, default="data/like_a1a.svm",
                             help=" path to dataset file in .svm format")
argument_parser.add_argument("--function_name", type=str, default="poisson_regression")

# optim_method_list = {'gradient', 'newton', 'hfn', 'bfgs', 'lbfgs', 'l1prox', 'sgd'}
optim_method_list = {'MOCK'}  # MOCK
argument_parser.add_argument("--optimize_method", type=str,
                             help=f"high-level optimization method, will be one of {optim_method_list}.")

# line_search_method = {'golden_search', 'brent', 'armijo', 'wolfe', 'lipschitz'}
line_search_method = {"MOCK"}  # MOCK. maybe should be only brent and set it like default
argument_parser.add_argument("--line_search",
                             type=str,
                             help=f"linear optimization method, will be one of {line_search_method}" +
                                  f"Note that you don't have to support a combination of 'newton' nor 'hfn' nor any 'bfgs' optimization method and 'lipschitz' linear search.")

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
