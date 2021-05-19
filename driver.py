import attr
import argparse
import json
import os
import random
import sys

import torch
from pprint import pprint

import model_config
from param import *
from train import train_tagger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ner')
    parser.add_argument('--model', type=str, default='simple')
    parser.add_argument('--config_name', type=str)
    parser.add_argument('--gpu_idx', type=int, default=-1)
    parser.add_argument("-o", "--overwrite", action='store_true')
    args = parser.parse_args()
    config: TaggingParams = getattr(model_config, args.config_name)(args.dataset, args.model)
    config.gpu_idx = args.gpu_idx
    pprint(attr.asdict(config))
    serialization_dir = f'models/{args.dataset}/{args.model}/{args.config_name}'
    random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])

    if not os.path.isdir(serialization_dir) or args.overwrite:
        if not os.path.isdir(serialization_dir):
            os.makedirs(serialization_dir)
        with open(f'{serialization_dir}/config.json', 'w') as f:
            f.write(json.dumps(attr.asdict(config), indent=4))
    else:
        sys.exit(f"{serialization_dir}, already exists. Please specify a new "
                 f"serialization directory or erase the existing one.")
    train_tagger(config=config, serialization_dir=serialization_dir, usecomet=False)
