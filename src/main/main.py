from src.main.args import argument_parser
from src.main.function.function import get_function

# MOCK
if __name__ == "__main__":

    args = argument_parser.parse_args()
    function = get_function(args.function_name)
    pass