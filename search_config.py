import os
import numpy as np
import pandas as pd
from hyperopt import hp
from pandas.core.algorithms import mode

from ray import tune
from ray.tune import Analysis

from param import *

datasets = ['ner', 'pos']
models = ['simple', 'crf']

glove = [f'embeddings/glove.twitter.27B.{dim}d.bin'
            for dim in [25, 50, 100, 200]]
word2vec = ['embeddings/word2vec_twitter_tokens.bin']
fasttext = ['embeddings/fasttext_twitter_raw.bin']
emb_paths = glove

dropout_options = [0, 0.05, 0.1, 0.2, 0.3]
lr_options = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1]

def search_SGD(dataset='ner', model='simple'):
    params = TaggingParams(dataset=dataset)
    if model == 'crf':
        params.model.type = 'neural_crf.NeuralCrf'
    params.training.optimizer = SGDOptimizerParams(
        type='torch.optim.SGD',
        lr=tune.loguniform(1e-5, 0.1),      # lr=0.05721092247217258
        momentum=tune.uniform(0.1, 0.9))    # momentum=0.6872224143884547
    params.is_grid = False
    return params

def get_search_name(dataset='ner', model='simple', emb=True, enc=1):
    name = ['search', dataset]
    if model == 'crf':
        name.append('crf')
    else:
        name.append('simple')

    if emb:
        name.append('emb')

    if enc == 1:
        name.append('enc1')
    elif enc == 2:
        name.append('enc2')

    return '-'.join(name)


def get_search_space(dataset='ner', model='simple', emb=True, enc=1):
    params = TaggingParams(dataset=dataset)
    name = ['search', dataset]
    if model == 'crf':
        params.model.type = 'neural_crf.NeuralCrf'
        name.append('crf')
    else:
        name.append('simple')

    if emb:
        params.model.embedding_param = tune.grid_search(emb_paths)
        name.append('emb')
    else:
        params.model.embedding_param = tune.grid_search(['25', '50'])

    if enc == 1:
        params.model.encoder = EncoderParams(
            type='torch.nn.LSTM',
            hidden_size=tune.grid_search([50, 100]),
            num_layers=1,
            dropout=0,
            bidirectional=tune.grid_search([True, False])
        )
        name.append('enc1')
    elif enc == 2:
        params.model.encoder = EncoderParams(
            type='torch.nn.LSTM',
            hidden_size=tune.grid_search([50, 100]),
            num_layers=2,
            # dropout=tune.uniform(0, 0.3),
            dropout=tune.grid_search(dropout_options),
            bidirectional=tune.grid_search([True, False])
        )
        name.append('enc2')
    params.training.optimizer = OptimizerParams(lr=tune.grid_search(lr_options))
    params.is_grid = True
    params.search_name = '-'.join(name)
    return params


def search_Adam(dataset='ner', model='simple'):
    params = TaggingParams(dataset=dataset)
    if model == 'crf':
        params.model.type = 'neural_crf.NeuralCrf'
    # params.training.optimizer = OptimizerParams(lr=tune.loguniform(1e-5, 0.1))      # lr=0.0029397976202716874
    params.training.optimizer = OptimizerParams(lr=tune.grid_search(lr_options))
    return params

def search_w_emb(dataset='ner', model='simple'):
    params = TaggingParams(dataset=dataset)
    if model == 'crf':
        params.model.type = 'neural_crf.NeuralCrf'
    params.model.embedding_param = tune.grid_search(['25', '50'] + emb_paths)
    # params.model.embedding_param = tune.grid_search(['50'])
    params.training.optimizer = OptimizerParams(lr=tune.grid_search(lr_options))
    # grid = 6
    return params

def search_w_enc1(dataset='ner', model='simple'):
    params = TaggingParams(dataset=dataset)
    if model == 'crf':
        params.model.type = 'neural_crf.NeuralCrf'
    params.model.embedding_param = tune.grid_search(['25', '50'] + emb_paths)
    params.model.encoder = EncoderParams(
        type='torch.nn.LSTM',
        hidden_size=tune.grid_search([50, 100]),
        num_layers=1,
        dropout=0,
        bidirectional=tune.grid_search([True, False])
    )
    params.training.optimizer = OptimizerParams(lr=tune.grid_search(lr_options))
    # grid = 6 * 3 * 2 * 2 = 72
    return params

def search_w_enc2(dataset='ner', model='simple'):
    params = TaggingParams(dataset=dataset)
    if model == 'crf':
        params.model.type = 'neural_crf.NeuralCrf'
    params.model.embedding_param = tune.grid_search(['25', '50'] + emb_paths)
    params.model.encoder = EncoderParams(
        type='torch.nn.LSTM',
        hidden_size=tune.grid_search([50, 100]),
        num_layers=2,
        # dropout=tune.uniform(0, 0.3),
        dropout=tune.grid_search(dropout_options),
        bidirectional=tune.grid_search([True, False])
    )
    params.training.optimizer = OptimizerParams(lr=tune.grid_search(lr_options))
    # grid = 6 * 3 * 2 * 2 = 72
    return params

def load_analysis(dataset, model, emb=False, enc=0, metric='val_loss'):
    mode = 'min' if metric == 'val_loss' else 'max'
    search_name = get_search_name(dataset, model, emb, enc)
    experiment_dir = f'ray_results/{search_name}'
    if not os.path.exists(experiment_dir):
        return None, None, None, None
    analysis = Analysis(experiment_dir=experiment_dir,
                        default_metric=metric, default_mode=mode)
    df: pd.DataFrame = analysis.dataframe()
    best_config = analysis.get_best_config()
    if metric == 'val_loss':
        best_results = df.loc[df['val_loss'].idxmin()].to_dict()
    else:
        best_results = df.loc[df['val_accuracy'].idxmax()].to_dict()
    param_cols = ['config/model.embedding_param']
    if enc != 0:
        param_cols.extend([
            'config/model.encoder.bidirectional',
            'config/model.encoder.hidden_size'
        ])
    if enc == 2:
        param_cols.append('config/model.encoder.dropout')
    param_cols.append('config/training.optimizer.lr')
    df = df[['val_accuracy'] + param_cols].sort_values(by=param_cols, ignore_index=True)
    return analysis, df, best_config, best_results

def table1(datasets=['ner', 'pos'],
           models=['simple', 'crf'],
           encs=[0, 1, 2],
           metric='val_loss'):
    columns = ['dataset', 'model', 'emb', 'enc', 'val_accuracy', 'embedding', 'lr']
    if min(encs) > 0:
        columns.extend(['bi', 'h_size'])
    if min(encs) > 1:
        columns.append(['dropout'])
    table_df = pd.DataFrame(columns=columns)
    for dataset in datasets:
        for model in models:
            for emb in [True, False]:
                for enc in encs:
                    search_name = get_search_name(dataset, model, emb, enc)
                    analysis, df, best_config, best_results = load_analysis(dataset, model, emb, enc, metric)
                    if analysis is None:
                        continue
                    best_val_accuracy = best_results['val_accuracy']
                    print(f'{search_name}: {best_val_accuracy}')
                    row = {
                        'dataset': dataset,
                        'model': model,
                        'emb': emb,
                        'enc': enc,
                        'embedding': best_config['model.embedding_param'],
                        'lr': best_config['training.optimizer.lr'],
                        'val_accuracy': best_val_accuracy
                    }
                    if min(encs) > 0:
                        row['bi'] = best_config['model.encoder.bidirectional']
                        row['h_size'] = best_config['model.encoder.hidden_size']
                    if min(encs) > 1:
                        row['dropout'] = best_config['model.encoder.dropout']
                    table_df = table_df.append(row, ignore_index=True)
    table_df = table_df.sort_values(by=['dataset', 'model', 'emb', 'enc'], ignore_index=True)
    return table_df
    # table_df = table_df.set_index(['Dataset', 'n'])
