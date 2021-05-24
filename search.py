import os
import pandas as pd
from pprint import pprint

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

import param
from models.neural_crf import NeuralCrf
from analysis.search_analysis import get_best_config
from param import ray_dir, valid_embeddings, TaggingParams, EncoderParams, AdamOptimizerParams
from util import get_search_name
from train import train_tagger

dropout_options = [0, 0.05, 0.1, 0.2, 0.3]
lr_options = [1e-3, 3e-3, 1e-2, 3e-2, 0.1]

grid_sizes = {
    (False, 0): 2 * len(lr_options),        # 10
    (True, 0): len(valid_embeddings) * len(lr_options),    # 20
    (False, 1): 2 * 2 * 2 * len(lr_options),    # 40
    (True, 1): len(valid_embeddings) * 2 * 2 * len(lr_options),    # 80
    (False, 2): 2 * 2 * 2 * len(dropout_options) * len(lr_options), # 200
    (True, 2): len(valid_embeddings) * 2 * 2 * len(dropout_options) * len(lr_options)  # 400
}

hyperopt_num_samples = {
    (False, 0): 10,
    (True, 0): 20,
    (False, 1): 40,
    (True, 1): 80,
    (False, 2): 100,
    (True, 2): 200,
}


def get_search_space(dataset='ner', model='simple', emb=True, enc=1, hyperopt=False):
    params = TaggingParams(dataset=dataset)

    if model == 'crf': params.model.type = NeuralCrf

    choices = tune.choice if hyperopt else tune.grid_search

    params.model.embedding_param = choices(valid_embeddings) if emb else choices(['25', '50'])

    if enc == 1:
        params.model.encoder = EncoderParams(
            type='torch.nn.LSTM',
            hidden_size=choices([50, 100]),
            num_layers=1,
            dropout=0,
            bidirectional=choices([True, False])
        )
    elif enc == 2:
        params.model.encoder = EncoderParams(
            type='torch.nn.LSTM',
            hidden_size=choices([50, 100]),
            num_layers=2,
            dropout=tune.grid_search(dropout_options) if not hyperopt else tune.uniform(0, 0.3),
            bidirectional=choices([True, False])
        )
    lr = tune.grid_search(lr_options) if not hyperopt else tune.loguniform(1e-3, 0.1)
    params.training.optimizer = AdamOptimizerParams(lr=lr)
    params.training.num_epochs = 30
    params.search_name = get_search_name(dataset, model, emb, enc, hyperopt=hyperopt)
    return params


# __main_begin__
def main(config: TaggingParams, name: str, num_samples=10, max_num_epochs=10,
         gpus_per_trial=1, checkpoint=True, usecomet=False, usehyperopt=False):
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=5,
        reduction_factor=2)
    print(config)
    pprint(config.to_flattened_dict())
    # Specify the search space and maximize score
    search_alg = None
    if usehyperopt:
        current_best_params = [get_best_config(args.dataset, args.model, args.emb, args.enc)]
        print(f'Using hyperopt with best guesses: {current_best_params}')
        search_alg = HyperOptSearch(metric="val_accuracy", mode="max",
                                    random_state_seed=config.random_seed,
                                    points_to_evaluate=current_best_params)
    else:
        num_samples = 1

    results = tune.run(
        tune.with_parameters(train_tagger, tuning=True, root_dir=os.getcwd(),
                             checkpoint=checkpoint, usecomet=usecomet),
        name=name,
        local_dir=ray_dir,
        resources_per_trial={"cpu": 2, 'gpu': gpus_per_trial},
        config=config.to_flattened_dict(),
        metric="val_loss",
        mode="min",
        num_samples=num_samples,
        stop={"training_iteration": max_num_epochs},
        scheduler=scheduler,
        search_alg=search_alg,
        log_to_file=True,
        verbose=1
    )
    df: pd.DataFrame = results.dataframe()
    best_config = results.get_best_config()
    best_results = df.loc[df['val_accuracy'].idxmax()].to_dict()

    print(f"Best trial config: {best_config}")
    print(f'Best trial final validation loss: {best_results["val_loss"]}')
    print(f'Best trial final validation accuracy: {best_results["val_accuracy"]}')
    # set_trace()
# __main_end__


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ner', choices=param.datasets)
    parser.add_argument('--model', type=str, default='simple', choices=param.model_types)
    parser.add_argument('--emb', action='store_true')
    parser.add_argument('--enc', type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('--gpu_idx', type=int, default=-1)
    parser.add_argument('--gpus_per_trial', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument("--smoke-test", action="store_true", help="Finish quickly for testing")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument('--usecomet', action='store_true')
    parser.add_argument('--usehyperopt', action='store_true')
    args, _ = parser.parse_known_args()

    config: TaggingParams = get_search_space(args.dataset, args.model, args.emb, args.enc,
                                             hyperopt=args.usehyperopt)
    config.gpu_idx = args.gpu_idx
    num_samples = hyperopt_num_samples[(args.emb, args.enc)] if args.usehyperopt else args.num_samples
    print(config.search_name)
    print(num_samples)
    # sys.exit()

    # shutdown currently running instance
    ray.shutdown()
    if args.smoke_test:
        ray.init(dashboard_host="0.0.0.0", num_cpus=2, local_mode=True)
        main(config, name=config.search_name,
             num_samples=1, max_num_epochs=config.training.num_epochs,
             gpus_per_trial=0, checkpoint=args.checkpoint)
    else:
        ray.init(dashboard_host="0.0.0.0")
        main(config, name=config.search_name, num_samples=num_samples,
             max_num_epochs=config.training.num_epochs, gpus_per_trial=args.gpus_per_trial,
             checkpoint=args.checkpoint, usecomet=args.usecomet, usehyperopt=args.usehyperopt)
