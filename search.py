import os
import sys
from pdb import set_trace
from comet_ml import Experiment

import torch
from torch.utils.data import DataLoader

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

import search_config
from param import TaggingParams
from util import load_object_from_dict, get_device
from train import load_datasets, train_tagger, test


# __main_begin__
def main(config: TaggingParams, name: str, num_samples=10,
         max_num_epochs=10, gpus_per_trial=1, checkpoint=True, usecomet=False):
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=5,
        reduction_factor=2)
    print(config)
    print(config.to_flattened_dict())
    # Specify the search space and maximize score
    search_alg = None
    if not config.is_grid:
        search_alg = HyperOptSearch(metric="val_accuracy", mode="max",
                                    random_state_seed=config.random_seed)
    else:
        num_samples = 1

    results = tune.run(
        tune.with_parameters(train_tagger, tuning=True, root_dir=os.getcwd(),
                             checkpoint=checkpoint, usecomet=usecomet),
        name=name,
        local_dir='ray_results/',
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

    best_trial = results.get_best_trial(metric="val_loss", mode="min", scope="all")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["val_loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["val_accuracy"]))

    if checkpoint:
        best_params = TaggingParams.from_flattened_dict(best_trial.config)
        train_dataset, validation_dataset = load_datasets(
            train_dataset_params=best_params.train_dataset,
            validation_dataset_params=best_params.validation_dataset
        )
        validation_dataloader = DataLoader(validation_dataset, best_params.val_batch_size)

        best_model = load_object_from_dict(best_params['model'],
                                        token_vocab=train_dataset.token_vocab,
                                        tag_vocab=train_dataset.tag_vocab)
        device = get_device(config.gpu_idx)
        best_model.to(device)
        checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
        best_model.load_state_dict(torch.load(checkpoint_path))

        test_acc = test(validation_dataloader, best_model, device)
        print("Best trial test set accuracy: {}".format(test_acc))
    # set_trace()
# __main_end__


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ner')
    parser.add_argument('--model', type=str, default='simple')
    parser.add_argument('--emb', action='store_true')
    parser.add_argument('--enc', type=int, default=0)
    parser.add_argument('--gpu_idx', type=int, default=-1)
    parser.add_argument('--gpus_per_trial', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument("--smoke-test", action="store_true", help="Finish quickly for testing")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument('--usecomet', action='store_true')
    args, _ = parser.parse_known_args()

    config: TaggingParams = search_config.get_search_space(args.dataset, args.model, args.emb, args.enc)
    config.gpu_idx = args.gpu_idx
    print(config.search_name)
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
        # Change this to activate training on GPUs
        main(config, name=config.search_name, num_samples=args.num_samples,
             max_num_epochs=config.training.num_epochs, gpus_per_trial=args.gpus_per_trial,
             checkpoint=args.checkpoint, usecomet=args.usecomet)