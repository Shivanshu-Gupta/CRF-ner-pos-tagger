import os
import json
import numpy as np
import pandas as pd
from copy import deepcopy

from param import *
from search_analysis import load_analysis
from util import get_model_name, get_configuration_name

def get_default_metrics(dataset='ner', model='simple'):
    serialization_dir = f'models/{dataset}/{model}-default'
    best_metrics = json.load(open(f'{serialization_dir}/best_metrics.json'))
    tst_metrics = json.load(open(f'{serialization_dir}/test_metrics.json'))
    best_metrics.update(tst_metrics)
    return best_metrics


def table_accuracies(datasets=['ner', 'pos'],
                     models=['simple', 'crf'],
                     encs=[0, 1, 2],
                     metric='val_accuracy'):
    columns = ['Dataset', 'Model Type', 'Configuration', 'Dev Accuracy', 'Test Accuracy']
    full_df = pd.DataFrame(columns=columns)
    for dataset in datasets:
        for model in models:
            default_metrics = get_default_metrics(dataset, model)
            row = {
                'Dataset': f'\textsc{{{dataset}}}',
                'Model Type': f'\textsc{{{model}}}',
                'Configuration': '\textsc{default}',
                'Dev Accuracy': default_metrics['val_accuracy'] * 100,
                'Test Accuracy': default_metrics['tst_accuracy'] * 100
            }
            full_df = full_df.append(row, ignore_index=True)
            for enc in encs:
                for emb in [False, True]:
                    serialization_dir = f'models/{dataset}/{get_model_name(model, emb, enc)}/'
                    if os.path.exists(f'{serialization_dir}/best_metrics.json'):
                        best_metrics = json.load(open(f'{serialization_dir}/best_metrics.json'))
                    else:
                        _, _, _, best_metrics = load_analysis(dataset, model, emb, enc, metric)
                    if best_metrics is None:
                        continue
                    best_val_accuracy = best_metrics['val_accuracy']
                    row = {
                        'Dataset': f'\textsc{{{dataset}}}',
                        'Model Type': f'\textsc{{{model}}}',
                        'Configuration': get_configuration_name(emb, enc, sep='+'),
                        'Dev Accuracy': best_val_accuracy * 100
                    }
                    if os.path.exists(f'{serialization_dir}/text_metrics.json'):
                        tst_metrics = json.load(open(f'{serialization_dir}/text_metrics.json'))
                        row['Test Accuracy'] = tst_metrics['tst_accuracy'] * 100

                    full_df = full_df.append(row, ignore_index=True)

    print(full_df)
    table_df = full_df[['Dataset', 'Model Type', 'Configuration', 'Dev Accuracy', 'Test Accuracy']]
    table_df = table_df.set_index(['Dataset', 'Model Type', 'Configuration'])
    table_df = table_df.unstack(1).reindex(index=pd.MultiIndex.from_product([['\textsc{ner}', '\textsc{pos}'],
                                                                             pd.unique(table_df.index.get_level_values(2))],
                                                                            names=['Dataset', 'Configuration']))
    table_df = table_df.sort_index(axis=1, ascending=False)[['Dev Accuracy', 'Test Accuracy']]
    print(table_df)
    print(table_df.to_latex(multirow=True, escape=False, float_format='%0.3f',
                            column_format='llcccc', multicolumn_format='c'))
    return table_df


def table_most_changed_labels(datasets=['ner', 'pos'], models=['simple', 'crf'],
                              encs=[0, 1, 2], metric='val_accuracy',
                              k=3, hyperopt=False):
    columns = ['Dataset', 'Model Type', 'Configuration', 'Most Changed Labels']
    table_df = pd.DataFrame(columns=columns)

    def get_most_changed_labels_str(acc0, acc1):
        per_tag_df = pd.DataFrame(columns=['tag', 'default', 'best'])
        tags = acc0.keys()
        for tag in tags:
            row = {
                'tag': tag,
                '0': acc0[tag] * 100,
                '1': acc1[tag] * 100
            }
            per_tag_df = per_tag_df.append(row, ignore_index=True)
            per_tag_df['diff'] = np.abs(per_tag_df['1'] - per_tag_df['0'])
            per_tag_df['improved'] = per_tag_df['1'] >= per_tag_df['0']
        most_changed_labels = per_tag_df.nlargest(k, 'diff').reset_index()[['tag', 'diff', 'improved']].to_numpy()
        most_changed_labels_str = []
        for t in most_changed_labels:
            if t[2]: most_changed_labels_str.append(f'{t[0]} +{{\small {t[1]:.2f}}}')
            else: most_changed_labels_str.append(f'{t[0]} -{{\small {t[1]:.2f}}}')
        return ', '.join(most_changed_labels_str)
    row = {}
    for dataset in datasets:
        row['Dataset'] = f'\textsc{{{dataset}}}'
        default_acc = get_default_metrics(dataset, 'simple')['val_accuracy_per_label']
        for model in models:
            row['Model Type'] = f'\textsc{{{model}}}'
            row['Configuration'] =  '\textsc{default}'
            if model == 'simple':
                row['Most Changed Labels'] = '-'
            else:
                row['Most Changed Labels'] = get_most_changed_labels_str(default_acc, get_default_metrics(dataset, model)['val_accuracy_per_label'])
            table_df = table_df.append(deepcopy(row), ignore_index=True)
            for enc in encs:
                for emb in [False, True]:
                    _, _, _, best_results = load_analysis(dataset, model, emb, enc, metric, hyperopt=hyperopt)
                    best_acc = {tag: best_results[f'val_accuracy_per_label/{tag}'] for tag in default_acc}
                    row['Configuration'] =  get_configuration_name(emb, enc, sep='+')
                    row['Most Changed Labels'] = get_most_changed_labels_str(default_acc, best_acc)
                    table_df = table_df.append(deepcopy(row), ignore_index=True)
    table_df = table_df.set_index(['Dataset', 'Model Type', 'Configuration'])
    print(table_df)
    print(table_df.to_latex(multirow=True, escape=False, column_format='llll'))
    return table_df
