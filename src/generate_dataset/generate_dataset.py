from src.generate_dataset.args import argument_parser, load_args_settings
from src.generate_dataset.dataset_processing import generate_dataset, store_dataset


if __name__ == "__main__":
    args = argument_parser.parse_args()
    load_args_settings(args)
    X, y, w = generate_dataset(args)
    store_dataset(X, y, w, args)
