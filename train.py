import copy
import datetime
import json
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from dataset import TwitterDataset, Vocabulary


def load_datasets(train_dataset_params: dict, validation_dataset_params: dict):
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


def train(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    serialization_dir: str
):
    start = time.time()
    best_metrics = {'validation_loss': 10e10}
    best_model = None
    for epoch_num in range(num_epochs):
        # training
        model.train()
        for batch in tqdm(train_dataloader, f'Epoch {epoch_num}'):
            optimizer.zero_grad()
            output_dict = model(**batch)
            output_dict['loss'].backward()
            optimizer.step()
        cur_epoch_metrics = model.get_metrics(header='train_')

        # compute validation metrics
        model.eval()
        for batch in validation_dataloader:
            model(**batch)
        cur_epoch_metrics.update(model.get_metrics(header='validation_'))

        # write the current epochs statistics to file
        with open(f'{serialization_dir}/metrics_epoch_{epoch_num}.json', 'w') as f:
            cur_epoch_metrics['epoch_num'] = epoch_num
            print(json.dumps(cur_epoch_metrics, indent=4))
            f.write(json.dumps(cur_epoch_metrics, indent=4))

        # check if current model is the best so far.
        if cur_epoch_metrics['validation_loss'] < best_metrics['validation_loss']:
            print('Best validation loss thus far...\n')
            best_model = copy.deepcopy(model)
            best_metrics = copy.deepcopy(cur_epoch_metrics)

    # write the best metrics we got and best model
    with open(f'{serialization_dir}/best_metrics.json', 'w') as f:
        best_metrics['run_time'] = str(datetime.timedelta(seconds=time.time()-start))
        print(f"Best Performing Model {json.dumps(best_metrics, indent=4)}")
        f.write(json.dumps(best_metrics, indent=4))
        torch.save(best_model, f'{serialization_dir}/model.pt')
