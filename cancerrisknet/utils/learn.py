import torch
from torch.utils import data
from collections import defaultdict
import cancerrisknet.learn.state_keeper as state
import cancerrisknet.models.factory as model_factory


def get_train_variables(args, model):
    """
        Given args, and whether or not resuming training, return
        relevant train variales.

        Returns:
            start_epoch:  Index of initial epoch
            epoch_stats: Dict summarizing epoch by epoch results
            state_keeper: Object responsibile for saving and restoring training state
            models: Dict of models
            optimizers: Dict of optimizers, one for each model
            tuning_key: Name of epoch_stats key to control learning rate by
            num_epoch_sans_improvement: Number of epochs since last dev improvment, as measured by tuning_key
    """
    start_epoch = 1
    args.lr = args.init_lr
    epoch_stats = init_metrics_dictionary()
    state_keeper = state.StateKeeper(args)

    # Set up models
    if isinstance(model, dict):
        models = model
    else:
        models = {args.model_name: model}

    # Setup optimizers
    optimizers = {}
    for name in models:
        model = models[name].to(args.device)

        optimizers[name] = model_factory.get_optimizer(model, args)

    num_epoch_sans_improvement = 0

    tuning_key = "dev_{}".format(args.tuning_metric)

    return start_epoch, epoch_stats, state_keeper, models, optimizers, tuning_key, num_epoch_sans_improvement


def concat_collate(batch):
    concat_batch = []
    for sample in batch:
        concat_batch.extend(sample)
    return data.dataloader.default_collate(concat_batch)


def init_metrics_dictionary():
    """
        An helper function. Return empty metrics dict.
    """
    stats_dict = defaultdict(list)
    stats_dict['best_epoch'] = 0
    return stats_dict


def get_dataset_loader(args, data):
    """
        Given args, and dataset class returns torch.utils.data.DataLoader
        Train/Dev/Attribution set are balanced. Test set is untouched. 
    """

    if data.split_group in ['train', 'dev', 'attribute']:
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
                weights=data.weights,
                num_samples=len(data),
                replacement=(data.split_group == 'train')
        )
        data_loader = torch.utils.data.DataLoader(
                data,
                num_workers=args.num_workers,
                sampler=sampler,
                pin_memory=True,
                batch_size=args.train_batch_size if data.split_group == 'train' else args.eval_batch_size,
                collate_fn=concat_collate
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            data,
            batch_size=args.train_batch_size if data.split_group == 'train' else args.eval_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=concat_collate,
            drop_last=False
        )

    return data_loader
