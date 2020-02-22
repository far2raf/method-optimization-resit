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
argument_parser.add_argument("--use_save_args_settings", action="store_true")

# Paths
argument_parser.add_argument("--args_settings_folder", type=str, default="src/generate_dataset/args_settings")
argument_parser.add_argument("--data_folder", type=str, default="data")

# Specific
argument_parser.add_argument("--number_features", default=2, type=int,
                             help="указывается с учетом фичи из одних 1. При 2, будет 1 рандомная фича, и одна из "
                                  "одних 1")
argument_parser.add_argument("--number_samples", default=2, type=int)
argument_parser.add_argument("--remove_bias", action="store_true")
