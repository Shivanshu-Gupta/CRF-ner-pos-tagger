import attr
import argparse
import json
import os
import random
import sys
from pandas.core.algorithms import mode

import torch
from pprint import pprint

import param
from model_config import get_best_tagger_params, get_default_tagger_params
from param import data_dir, models_dir, TaggingParams
from train import train_tagger, test
from dataset import TwitterDataset
from util import get_device, get_model_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ner', choices=param.datasets)
    parser.add_argument('--model', type=str, default='simple', choices=param.model_types)
    parser.add_argument('--default', action='store_true')
    parser.add_argument('--emb', action='store_true')
    parser.add_argument('--enc', type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('--gpu_idx', type=int, default=-1)
    parser.add_argument("-o", "--overwrite", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--silent", action='store_true')
    args = parser.parse_args()
    if args.default:
        model_name = f'{args.model}-default'
        config: TaggingParams = get_default_tagger_params(args.dataset, args.model, args.emb, args.enc)
    else:
        model_name = get_model_name(args.model, args.emb, args.enc)
        config: TaggingParams = get_best_tagger_params(args.dataset, args.model, args.emb, args.enc)
    config.gpu_idx = args.gpu_idx
    pprint(attr.asdict(config))

    serialization_dir = os.path.join(models_dir, args.dataset, model_name)
    print(serialization_dir)
    # sys.exit()
    if os.path.isdir(serialization_dir) and not args.overwrite:
        sys.exit(f"{serialization_dir}, already exists. Please specify a new "
                 f"serialization directory or erase the existing one.")
    elif not os.path.isdir(serialization_dir):
        os.makedirs(serialization_dir)

    with open(f'{serialization_dir}/config.json', 'w') as f:
        f.write(json.dumps(attr.asdict(config), indent=4))

    random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])
    train_tagger(config=config, serialization_dir=serialization_dir,
                 usecomet=False, silent=args.silent)
    if args.test:
        model = torch.load(f"{serialization_dir}/model.pt").eval()
        dataset = TwitterDataset(os.path.join(data_dir, f'twitter_test.{args.dataset}'))
        dataset.set_vocab(model.token_vocab, model.tag_vocab)
        test(tst_loader=torch.utils.data.DataLoader(dataset, 1),
             model=model, device=get_device(config.gpu_idx),
             serialization_dir=serialization_dir)
