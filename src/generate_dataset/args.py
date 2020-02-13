import argparse
import json
import os


def load_args_settings(args):
    argparse_dict = vars(args)
    path = os.path.abspath(f"{args.args_settings_folder}/{args.function_name}.json")
    _json = json.load(open(path, "r"))
    argparse_dict.update(_json)


argument_parser = argparse.ArgumentParser()

# General
function_names_list = ["poisson_regression", "linear"]
argument_parser.add_argument("--function_name", type=str, choices=function_names_list)
argument_parser.add_argument("--seed", type=int, default=42)

# Paths
argument_parser.add_argument("--args_settings_folder", type=str, default="src/generate_dataset/args_settings")
argument_parser.add_argument("--data_folder", type=str, default="data")

# Specific
argument_parser.add_argument("--number_features", type=int)
argument_parser.add_argument("--number_samples", type=int)
argument_parser.add_argument("--remove_bias", action="store_true")
