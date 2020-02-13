import scipy
from sklearn import datasets
import numpy as np


def load_dataset(args):
    path = f"{args.data_folder}/{args.function_name}"
    X, y = datasets.load_svmlight_file(f"{path}.svm")
    w = np.load(f"{path}.npy")
    return X, y, w


def store_dataset(X, y, w, args):
    path = f"{args.data_folder}/{args.function_name}"
    datasets.dump_svmlight_file(X, y.squeeze(), f"{path}.svm")
    np.save(f"{path}.npy", w)


def generate_X(args):
    if args.remove_bias:
        X = scipy.sparse.rand(args.number_samples, args.number_features, density=0.1)  # Size: (num_samples, num_features)
    else:
        X = scipy.sparse.rand(args.number_samples, args.number_features - 1, density=0.1)  # Size: (num_samples, num_features)
        ones = np.ones((args.number_samples, 1))
        X = scipy.sparse.hstack((X, ones))
    return X


def generate_dataset_poisson_regression(args):
    """
        Generate dataset poisson regression with size like a1a dataset
    """
    X = generate_X(args)  # Size: (num_samples, num_features)
    w = np.random.rand(args.number_features, 1)
    mean_y = X.dot(w)
    y = np.random.poisson(lam=mean_y)  # Size: (number_samples, 1)
    return X, y, w


def generate_dataset_linear(args):
    """
        Generate simple dataset with linear
    """
    X = generate_X(args)  # Size: (num_samples, num_features)
    w = np.random.rand(args.number_features, 1)
    y = X.dot(w)
    return X, y, w


def generate_dataset(args):
    if args.function_name == "poisson_regression":
        X, y, w = generate_dataset_poisson_regression(args)
    elif args.function_name == "linear":
        X, y, w = generate_dataset_linear(args)
    else:
        raise RuntimeError(f"The function name doesn't determined: {args.function_name}")
    return X, y, w
