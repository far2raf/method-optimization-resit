from src.generate_dataset.dataset_processing import load_dataset
from src.main.args import argument_parser
from src.main.function.function import get_function

if __name__ == "__main__":

    args = argument_parser.parse_args()
    function = get_function(args)
    X, y, w = load_dataset(args)
    # method = get_method(args)
    pass
