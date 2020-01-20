# MOCK. new file. not finished
import scipy
from sklearn import datasets
import numpy as np


def generate_dataset_poisson_regression(args):
    w = np.random.rand(args.number_features, 1)
    X = scipy.sparse.rand(args.number_samples, args.number_features, density=0.1)
    before_y = X.dot(w)
    y = np.random.poisson(lam=before_y) # (number_samples, 1)
    datasets.dump_svmlight_file(X, y.squeeze(), args.ds_path)
    np.save(args.ds_answer_path, y)


def generate_dataset(args):
    if args.function_name == "poisson_regression":
        generate_dataset_poisson_regression(args)
    else:
        raise RuntimeError(f"The function name doesn't determined: {args.function_name}")


#MOCK. should be deleted
def test(args):
    data = datasets.load_svmlight_file(args.ds_path_test)
    return data


if __name__ == "__main__":
    from args import argument_parser
    from args import load_args_settings

    args = argument_parser.parse_args()
    load_args_settings(args)
    generate_dataset(args)
