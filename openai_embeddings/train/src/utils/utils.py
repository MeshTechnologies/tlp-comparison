import itertools
import yaml
from pathlib import Path
import os
import json


def flatten_grid(input_dict):  # creates list of experiments for grid test
    for key, val in input_dict.items():
        if not isinstance(val, list):
            input_dict[key] = [val]

    keys, vals = zip(*input_dict.items())
    combos = [dict(zip(keys, v)) for v in itertools.product(*vals)]
    return combos


def get_config_yaml():
    path = Path(__file__).parent.absolute()  # path of file being run
    exp_yaml_path = "/".join(str(path).split('/')[:-1]) + "/config.yaml"
    with open(exp_yaml_path) as yaml_data:
        exp_config = yaml.load(yaml_data, Loader=yaml.FullLoader)

    return exp_config, exp_yaml_path  # return path to log yaml to comet


def f1_score(precision, recall):
    return 2 * ((precision * recall) / (precision + recall))


def _listdir(path):
    return [item for item in os.listdir(path) if item != ".DS_Store"]


def _load_json(path):
    with open(path, "r") as fp:
        d = json.load(fp)
    return d


def _dump_json(d, path):
    with open(path, "w") as fp:
        json.dump(d, fp)


def _load_yaml(path):
    with open(path, "r") as fp:
        d = yaml.load(fp, Loader=yaml.FullLoader)
    return d


def _dump_yaml(d, path):
    with open(path, 'w') as fp:
        yaml.dump(d, fp)


def script_path(filename):  # from metalflow tutorial #1
    filepath = os.path.join(os.path.dirname(__file__))
    return os.path.join(filepath, filename)


def get_hyperparameter_values(hyperparams):
    H_out = {}
    for hyperparam in hyperparams:
        H_out[hyperparam] = hyperparams[hyperparam]["value"]
    return H_out
