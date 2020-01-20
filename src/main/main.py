# MOCK
if __name__ == "__main__":
    from args import argument_parser
    from function.function import get_function

    args = argument_parser.parse_args()
    function = get_function(args.function_name)
    pass
