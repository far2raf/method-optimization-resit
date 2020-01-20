import argparse
import json


def load_args_settings(args):
    argparse_dict = vars(args)
    _json = json.load(open(args.args_settings_path, "r"))
    argparse_dict.update(_json)

argument_parser = argparse.ArgumentParser()

# Common
argument_parser.add_argument("--ds_path_test", type=str, default="../../data/a1a.t") # MOCK. should be deleted
argument_parser.add_argument("--args_settings_path", type=str, default="args_settings/like_a1a.json")

# Specific
argument_parser.add_argument("--ds_path", type=str,
                             help=" path to dataset file in .svm format")
argument_parser.add_argument("--ds_answer_path", type=str)
argument_parser.add_argument("--number_features", type=int)
argument_parser.add_argument("--number_samples", type=int)
argument_parser.add_argument("--function_name", type=str)
argument_parser.add_argument("--seed", type=int)

