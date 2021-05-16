import attr
import argparse
import json
import os
import random
import sys

import torch
from pprint import pprint
from torch.utils.data import DataLoader

from param import params
from train import load_datasets, train
from util import load_object_from_dict

USE_PARAM = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    if not USE_PARAM:
        parser.add_argument("config_path", help="path to configuration file")
    parser.add_argument("-s", "--serialization_dir", required=True,
                        help="save directory for model, dataset, and metrics")
    parser.add_argument("-o", "--overwrite", action='store_true')
    args = parser.parse_args()
    if not USE_PARAM:
        config = json.load(open(args.config_path))
        pprint(config)
    else:
        config = params
        pprint(attr.asdict(config))
    serialization_dir = args.serialization_dir
    random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])

    if not os.path.isdir(serialization_dir) or args.overwrite:
        if not os.path.isdir(serialization_dir):
            os.makedirs(serialization_dir)
        with open(f'{serialization_dir}/config.json', 'w') as f:
            if not USE_PARAM:
                f.write(json.dumps(config, indent=4))
            else:
                f.write(json.dumps(attr.asdict(config), indent=4))
    else:
        sys.exit(f"{serialization_dir}, already exists. Please specify a new "
                 f"serialization directory or erase the existing one.")
    # sys.exit()
    # load PyTorch `Dataset` and `DataLoader` objects
    train_dataset, validation_dataset = load_datasets(
        train_dataset_params=config['train_dataset'],
        validation_dataset_params=config['validation_dataset']
    )
    batch_size = config['training']['batch_size']
    train_dataloader = DataLoader(train_dataset, batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size)

    # load model
    model = load_object_from_dict(config['model'],
                                  token_vocab=train_dataset.token_vocab,
                                  tag_vocab=train_dataset.tag_vocab)

    # load optimizer
    optimizer = load_object_from_dict(config['training']['optimizer'],
                                      params=model.parameters())

    train(
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        optimizer=optimizer,
        num_epochs=config['training']['num_epochs'],
        serialization_dir=serialization_dir
    )
