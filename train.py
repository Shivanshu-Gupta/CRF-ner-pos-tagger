import os
import copy
import random
import datetime
import json
import time
from tqdm import tqdm
from typing import Union

from comet_ml import Experiment

import torch
from torch.utils.data import DataLoader

from ray import tune

from param import TaggingParams, DataParams
from dataset import TwitterDataset, Vocabulary
from util import load_object_from_dict, output, get_device


def load_datasets(train_dataset_params: Union[dict, DataParams],
                  validation_dataset_params: Union[dict, DataParams]):
    # load PyTorch ``Dataset`` objects for the train & validation sets
    train_dataset = TwitterDataset(**train_dataset_params)
    validation_dataset = TwitterDataset(**validation_dataset_params)

    # use tokens and tags in the training set to create `Vocabulary` objects
    token_vocab = Vocabulary(train_dataset.get_tokens_list(), add_unk_token=True)
    tag_vocab = Vocabulary(train_dataset.get_tags_list())

    # add `Vocabulary` objects to datasets for tokens/tags to ID mapping
    train_dataset.set_vocab(token_vocab, tag_vocab)
    validation_dataset.set_vocab(token_vocab, tag_vocab)

    return train_dataset, validation_dataset


def train(train_dataloader: DataLoader,
          model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          device) -> dict:
    model.train()
    # for batch in tqdm(train_dataloader, f'Epoch {epoch_num}'):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        output_dict = model(**batch)
        output_dict['loss'].backward()
        optimizer.step()
    return model.get_metrics(header='trn_')


def eval(validation_dataloader: DataLoader,
         model: torch.nn.Module,
         device) -> dict:
    model.eval()
    for batch in validation_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        model(**batch)
    return model.get_metrics(header='val_')

def run_train_epoch(epoch_num, model, trn_loader, val_loader, optimizer, device, experiment=None):
    curr_epoch_metrics = train(trn_loader, model, optimizer, device)
    curr_epoch_metrics.update(eval(val_loader, model, device))
    if experiment is not None:
        experiment.log_metrics({k: v for k, v in curr_epoch_metrics.items() if not k.endswith('label')},
                                step=epoch_num)
    return curr_epoch_metrics


def train_w_tuning(checkpoint=False, experiment=None, **train_params):
    num_epochs = train_params['num_epochs']
    train_params.pop('num_epochs')
    for epoch_num in range(num_epochs):
        curr_epoch_metrics = run_train_epoch(epoch_num, experiment=experiment, **train_params)
        tune.report(**curr_epoch_metrics)
        if checkpoint:
            # Save checkpoint
            with tune.checkpoint_dir(step=epoch_num) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save(train_params['model'].state_dict(), path)

def train_wo_tuning(serialization_dir=None, experiment=None, **train_params):
    start = time.time()
    best_metrics = {'val_loss': 10e10}
    best_model = None
    num_epochs = train_params['num_epochs']
    train_params.pop('num_epochs')
    for epoch_num in range(num_epochs):
        curr_epoch_metrics = run_train_epoch(epoch_num, experiment=experiment, **train_params)
        print(curr_epoch_metrics)
        # write the current epochs statistics to file
        curr_epoch_metrics['epoch_num'] = epoch_num
        output(json.dumps(curr_epoch_metrics, indent=4),
            filepath=f'{serialization_dir}/metrics_epoch_{epoch_num}.json')

        # check if current model is the best so far.
        if curr_epoch_metrics['val_loss'] < best_metrics['val_loss']:
            print('Best validation loss thus far...\n')
            best_model = copy.deepcopy(train_params['model'])
            best_metrics = copy.deepcopy(curr_epoch_metrics)

    # write the best metrics we got and best model
    best_metrics['run_time'] = str(datetime.timedelta(seconds=time.time() - start))
    output(f"Best Performing Model {json.dumps(best_metrics, indent=4)}",
            filepath=f'{serialization_dir}/best_metrics.json')
    torch.save(best_model, f'{serialization_dir}/model.pt')


def train_tagger(config: Union[dict, TaggingParams],
                 tuning: bool = False,
                 checkpoint=True,
                 serialization_dir: str = None,
                 checkpoint_dir: str =None,
                 root_dir: str = None,
                 usecomet: bool = False):
    if tuning:
        os.chdir(root_dir)
    print(config)

    if usecomet:
        experiment = Experiment(project_name=config['dataset'])
        if config['search_name']:
            experiment.add_tag(config['search_name'].split('-')[1])
        if isinstance(config, TaggingParams):
            experiment.log_parameters(config.to_flattened_dict())
        else:
            experiment.log_parameters(config)

    if not isinstance(config, TaggingParams):
        config: TaggingParams = TaggingParams.from_flattened_dict(config)
    random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])

    # load PyTorch `Dataset` and `DataLoader` objects
    train_dataset, validation_dataset = load_datasets(
        train_dataset_params=config.train_dataset,
        validation_dataset_params=config.validation_dataset
    )
    trn_loader = DataLoader(train_dataset, config.train_batch_size)
    val_loader = DataLoader(validation_dataset, config.val_batch_size)

    # load model
    model = load_object_from_dict(config['model'],
                                  token_vocab=train_dataset.token_vocab,
                                  tag_vocab=train_dataset.tag_vocab)

    # load optimizer
    optimizer = load_object_from_dict(config['training']['optimizer'],
                                      params=model.parameters())
    device = get_device(config.gpu_idx)
    model.to(device)

    num_epochs = config.training.num_epochs
    train_params = {
        'model': model,
        'trn_loader': trn_loader,
        'val_loader': val_loader,
        'optimizer': optimizer,
        'num_epochs': num_epochs,
        'device': device
    }
    if usecomet:
        with experiment.train():
            if tuning: train_w_tuning(checkpoint=checkpoint, experiment=experiment, **train_params)
            else: train_wo_tuning(serialization_dir=serialization_dir, experiment=experiment, **train_params)
    else:
        if tuning: train_w_tuning(checkpoint=checkpoint, **train_params)
        else: train_wo_tuning(serialization_dir=serialization_dir, **train_params)



def test(test_dataloader: DataLoader,
         model: torch.nn.Module,
         device) -> dict:
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)
    return model.get_metrics(header='tst_')
