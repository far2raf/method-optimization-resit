import numpy as np

from src.generate_dataset.dataset_processing import load_dataset
from src.main.args import argument_parser, load_args_settings
from src.main.functions.common import get_function
from src.main.optim_methods import get_opt_method_maker
from src.main.stop_conditions.common import get_stop_condition

if __name__ == "__main__":

    args = argument_parser.parse_args()
    if not args.no_use_save_args_settings:
        load_args_settings(args)
    np.random.seed(args.seed)

    X, y, w = load_dataset(args)
    S, F = X.shape
    y = y.reshape((S, 1))
    assert y.shape == (S, 1)
    assert w.shape == (F, 1)

    function = get_function(args)
    stop_condition = get_stop_condition(args, F)
    opt_method_maker = get_opt_method_maker(args)
    opt_method = opt_method_maker(X, y.reshape((-1, 1)), function, stop_condition)
    answer = opt_method.run()

    print(f"Sum of square diff: {round(((answer.get_optimal_point() - w) ** 2).sum(), 6)}")
    print(f"Loss: {round(function.loss(answer.get_optimal_point(), X, y).item(), 6)}")
    # print(f"Diff between true and optim: \n {answer.get_optimal_point() - w}")
