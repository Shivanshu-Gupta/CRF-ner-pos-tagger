import os
import pandas as pd

from ray.tune import Analysis

from param import *
from util import get_search_name


def load_analysis(dataset, model, emb=False, enc=0, metric='val_loss', hyperopt=False):
    mode = 'min' if metric == 'val_loss' else 'max'
    search_name = get_search_name(dataset, model, emb, enc, hyperopt=hyperopt)
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


def df_search_results(datasets=['ner', 'pos'], models=['simple', 'crf'],
                          encs=[0, 1, 2], metric='val_accuracy',
                          add_param_cols=True, hyperopt=False):
    columns = ['Dataset', 'model', 'emb', 'enc', 'Dev Accuracy']
    if add_param_cols:
        columns.extend(['embedding', 'lr'])
        columns.extend(['bi', 'h_size'])
        columns.append('dropout')
    table_df = pd.DataFrame(columns=columns)
    for dataset in datasets:
        for model in models:
            for enc in encs:
                for emb in [False, True]:
                    search_name = get_search_name(dataset, model, emb, enc, hyperopt=hyperopt)
                    _, _, best_config, best_results = load_analysis(dataset, model, emb, enc, metric, hyperopt=hyperopt)
                    if best_config is None:
                        continue
                    best_val_accuracy = best_results['val_accuracy']
                    print(f'{search_name}: {best_val_accuracy}')
                    row = {
                        'Dataset': f'\textsc{{{dataset}}}',
                        'model': model,
                        'emb': emb,
                        'enc': enc,
                        'Dev Accuracy': best_val_accuracy * 100
                    }
                    if add_param_cols:
                        row['embedding'] = best_config['model.embedding_param']
                        row['lr'] = best_config['training.optimizer.lr']
                        if best_config['model.encoder.type']:
                            row['bi'] = best_config['model.encoder.bidirectional']
                            row['h_size'] = best_config['model.encoder.hidden_size']
                        if best_config['model.encoder.num_layers'] > 1:
                            row['dropout'] = best_config['model.encoder.dropout']
                    table_df = table_df.append(row, ignore_index=True)
    print(table_df)
    return table_df


def get_best_config(dataset, model, emb, enc, metric='val_accuracy', structure=False, hyperopt=False):
    _, _, best_config, _ = load_analysis(dataset, model, emb, enc, metric=metric, hyperopt=hyperopt)
    if best_config is None: return None
    elif structure: return TaggingParams.from_flattened_dict(best_config)
    else: return best_config


def get_best_configs(datasets=['ner', 'pos'], models=['simple', 'crf'],
                     encs=[0, 1, 2], metric='val_accuracy', hyperopt=False):
    best_configs = {}
    for dataset in datasets:
        for model in models:
            for emb in [True, False]:
                for enc in encs:
                    best_config = get_best_config(dataset, model, emb, enc, metric, structure=True, hyperopt=hyperopt)
                    if best_config is None:
                        continue
                    best_configs[(dataset, model, emb, enc)] = best_config
    return best_configs
