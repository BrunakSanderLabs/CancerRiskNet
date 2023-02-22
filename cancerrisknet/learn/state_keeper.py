import pickle
import os
import torch
import collections
import hashlib
import copy

OPTIMIZER_PATH = '{}_{}_optim.pt'
MODEL_PATH = '{}_{}_model.pt'
PARAM_PATH = '{}_{}_param.p'
STATS_PATH = '{}_{}_stats.p'
ERROR_MSG = "Sorry, {} does not exist!"


def md5(key):
    """
        Returns a md5 hashed string
    """
    return hashlib.md5(key.encode()).hexdigest()


def get_model_path(args):
    model_dir = args['model_dir'] if type(args) == dict else args.model_dir
    return os.path.join(model_dir, MODEL_PATH.format(args.model_name, args.exp_id))


class StateKeeper:
    """Takes care of saving and loading models for resumable training."""
    def __init__(self, args):
        self.args = args
        self.identifier = args.exp_id
        self.model_name = args.model_name

    def save(self, models, optimizers, epoch, lr, epoch_stats):
        """
        Save the state of a run to be loaded later in case it will ger resumed.
        Args:
            models: dictionary of torch models used in the run
            optimizers: dictionary of optimizers used, must correspond one to one with models
            epoch: an integer representing the epoch the run is at
            lr: current learning rate of the optimizer
            epoch_stats: current stats for the run
        """
        # Save dict for epoch and lr.
        param_dict = {'epoch': epoch, 'lr': lr}
        identifier = self.identifier
        os.makedirs(self.args.model_dir, exist_ok=True)
        param_path = os.path.join(self.args.model_dir, PARAM_PATH.format(self.args.model_name, identifier))
        with open(param_path, 'wb') as param_file:
            pickle.dump(param_dict, param_file)

        # Save epoch_stats dict.
        stats_path = os.path.join(self.args.model_dir, STATS_PATH.format(self.args.model_name, identifier))
        with open(stats_path, 'wb') as stats_file:
            pickle.dump(epoch_stats, stats_file)

        model_paths = []
        for model_name in models:
            # save model
            model = models[model_name]
            model_path = os.path.join(self.args.model_dir, MODEL_PATH.format(model_name, identifier))

            torch.save(model, model_path)
            # save optimizer
            optimizer = optimizers[model_name]
            optimizer_path = os.path.join(self.args.model_dir, OPTIMIZER_PATH.format(model_name, identifier))
            torch.save(optimizer.state_dict(), optimizer_path)

            model_paths.append(model_path)
        return model_paths

    def load(self):
        """
            Loads the state of a run to resume based on the arguments specified.

            Returns:
                models: a dictionary of the torch models to use in the run to resume
                optimizer_states: a dictionary of the optimizer states to use in the
                                  run to resume. One is assumed to exist for each model
                epoch: an integer representing the epoch to resume from
                lr: current learning rate to start from
                epoch_stats: current stats for the run
        """
        identifier = self.identifier
        # Load dict for epoch and lr.
        param_path = os.path.join(self.args.model_dir, PARAM_PATH.format(self.args.model_name, identifier))

        try:
            with open(param_path, 'rb') as param_file:
                param_dict = pickle.load(param_file)
        except Exception as e:
            print(e.message)

        # Load epoch_stats dict.
        stats_path = os.path.join(self.args.model_dir, STATS_PATH.format(self.args.model_name, identifier))
        try:
            with open(stats_path, 'rb') as stats_file:
                epoch_stats = pickle.load(stats_file)
        except Exception as e:
            print(e.message)

        # Load model and corresponding optimizers.
        models = {}
        optimizer_states = {}
        model_names = [self.model_name]

        for model_name in model_names:
            # Load model
            model_path = os.path.join(self.args.model_dir, MODEL_PATH.format(model_name, identifier))
            try:
                models[model_name] = torch.load(model_path, map_location=self.args.device)
            except Exception:
                raise Exception(
                    ERROR_MSG.format(model_path))
            print("Loading from " + str(model_path))
            # Load optimizer state
            optimizer_path = os.path.join(self.args.model_dir, OPTIMIZER_PATH.format(model_name, identifier))
            try:
                optimizer_states[model_name] = torch.load(optimizer_path, map_location=self.args.device)
            except Exception:
                raise Exception(
                    ERROR_MSG.format(optimizer_path))

        return models, optimizer_states, param_dict['epoch'], param_dict['lr'], epoch_stats

    def load_optimizer(self, optimizer, state_dict):
        """
            Given an optimizer and a state_dict, loads the state_dict into
            the optimizer while preserving correct device placement.

            Returns:
                optimizer, with new state_dict

        """
        # Build mapping from param to device
        param_to_device = {}
        for param_key in state_dict['state']:
            param = state_dict['state'][param_key]
            for attribute_key in param:
                if isinstance(param[attribute_key], int) or isinstance(param[attribute_key], float):
                    continue
                param_to_device["{}_{}".format(param_key, attribute_key)] = param[attribute_key].get_device()

        optimizer.load_state_dict(state_dict)
        if self.args.cuda:
            # Move params to correct gpus. Load_state_dict uses copy.deepcopy which loses device information
            for param_key in optimizer.state_dict()['state']:
                param = optimizer.state_dict()['state'][param_key]
                for attribute_key in param:
                    if isinstance(param[attribute_key], int) or isinstance(param[attribute_key], float):
                        continue
                    optimizer.state_dict()['state'][param_key][attribute_key] = \
                        optimizer.state_dict()['state'][param_key][attribute_key].cuda(
                            param_to_device["{}_{}".format(param_key, attribute_key)]
                        )

        return optimizer
