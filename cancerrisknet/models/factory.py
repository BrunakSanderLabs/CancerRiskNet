import torch
from torch import nn
import cancerrisknet.learn.state_keeper as state

MODEL_REGISTRY = {}
NO_MODEL_ERR = 'Model {} not in MODEL_REGISTRY! Available models are {} '
NO_OPTIM_ERR = 'Optimizer {} not supported!'


def RegisterModel(model_name):
    """Registers a configuration."""

    def decorator(f):
        MODEL_REGISTRY[model_name] = f
        return f

    return decorator


def get_model(args):
    return get_model_by_name(args.model_name, args)


def get_model_by_name(name, args):
    """
        Get model from MODEL_REGISTRY based on args.model_name
        Args:
            name: Name of model, must exit in registry
            args: run ime args from parsing

        Returns:
            model: an instance of some torch.nn.Module
    """
    if name not in MODEL_REGISTRY:
        raise Exception(
            NO_MODEL_ERR.format(
                name, MODEL_REGISTRY.keys()))
    model = MODEL_REGISTRY[name](args)
    return model


def load_model(path, args):
    print('\nLoading model from [%s]...' % path)

    model = torch.load(path)
    if isinstance(model, dict):
        model = model[args.model_name]

    if isinstance(model, nn.DataParallel):
        model = model.module.cpu()

    return model


def get_params(model):
    """
      Helper function to get parameters of a model.
    """

    return model.parameters()


def get_optimizer(model, args):
    """
        Helper function to fetch optimizer based on args.
    """
    params = [param for param in model.parameters() if param.requires_grad]
    if args.optimizer == 'adam':
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adagrad':
        return torch.optim.Adagrad(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        return torch.optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise Exception(NO_OPTIM_ERR.format(args.optimizer))
