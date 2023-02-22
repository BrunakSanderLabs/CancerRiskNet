import hashlib
import torch
import argparse
import pandas as pd
import sys
import pickle
import cancerrisknet.learn.state_keeper as state
import yaml
import os
import warnings


POSS_VAL_NOT_LIST = 'Flag {} has an invalid list of values: {}. Length of slist must be >=1'


def parse_args(args_str=None):
    parser = argparse.ArgumentParser(description='CancerRiskNet Classifier')
    # What main steps to execute
    parser.add_argument('--train', action='store_true', default=False, help='Whether or not to train model')
    parser.add_argument('--test', action='store_true', default=False, help='Whether or not to run model on test set')
    parser.add_argument('--dev', action='store_true', default=False, help='Whether or not to run model on dev set')
    parser.add_argument('--attribute', action='store_true', default=False,
                        help='Whether or not to run attribution analysis (interpretation). '
                             'Attribution is performed on test set')

    # Device specification
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Enable the gpu computation. If enabled but no CUDA is found then keep using CPU.')
    parser.add_argument('--num_workers', type=int, default=8, help='num workers for each data loader [default: 8]')

    # Dataset setup
    parser.add_argument('--dataset', type=str, default='disease_progression',
                        help="Name of dataset to use. Default: 'disease_preogression")
    parser.add_argument('--metadata_path', type=str, default='data/metadata.json', help="Path of json source datafile")
    parser.add_argument('--data_setting_path', type=str, default='data/settings.yaml',
                        help="Path of yaml with data specific settings")
    parser.add_argument('--month_endpoints', nargs='+', default=[3, 6, 12, 36, 60, 120],
                        help="List of month endpoints at which to generate risk prediction.")
    parser.add_argument('--pad_size', type=int, default=None,
                        help="Padding the trajectories to how long for training. Default: Pad every trajectory to "
                             "the max_events_length.")
    parser.add_argument('--icd10_level', type=int, default=3,
                        help="Which level of ICD10 code is used. Default: 3, i.e. C25.")
    parser.add_argument('--icd8_level', type=int, default=3,
                        help="Which level of ICD8 code is used. Default: 3, i.e. 123.")
    parser.add_argument('--max_events_length', type=int, default=300,
                        help="Max num of events to use. Apply a n-gram frame shift if exceeded.")
    parser.add_argument('--min_events_length', type=int, default=5, help="Min num of events to include a patient")
    parser.add_argument('--exclusion_interval', type=int, default=0,
                        help="Exclude events before end of trajectory, default: 0 (month).")
    parser.add_argument('--use_known_risk_factors_only', action='store_true', default=False,
                        help="Whether to use only known risk factors for training.")
    parser.add_argument('--no_random_sample_eval_trajectories', action='store_true', default=False,
                        help="Whether False trajectories are sampled randomly from each patient during dev and test. "
                             "If no_random_sample_eval_trajectories=True, use all the trajectories.")
    parser.add_argument('--max_eval_indices', type=int, default=250,
                        help="Max number of trajectories to include for each patient during dev and test. ")

    # Hyper-params for model training
    parser.add_argument('--model_name', type=str, default='transformer', help="Model to be used.")
    parser.add_argument('--num_layers', type=int, default=1, help="Number of layers to use for sequential NNs.")
    parser.add_argument('--num_heads', type=int, default=None,
                        help="Number of heads to use for multihead attention. Only relevant for transformer.")
    parser.add_argument('--hidden_dim', type=int, default=256, help="Representation size at end of network.")
    parser.add_argument('--pool_name', type=str, default='GlobalAvgPool',
                        help='Pooling mechanism. Choose from ["Softmax_AttentionPool","GlobalAvgPool","GlobalMaxPool"]')
    parser.add_argument('--dropout', type=float, default=0.2, help="Dropout value for the neural network model.")
    parser.add_argument('--use_time_embed', action='store_true', default=False,
                        help='Whether or not to condition embeddings by their relative time to the outcome date.')
    # TODO: New feature request: we should have another absolute time input to offset the changes of clinical practice
    parser.add_argument('--use_age_embed', action='store_true', default=False,
                        help='Whether or not to condition embeddings by the age at administration.')
    parser.add_argument('--add_age_neuron', action='store_true', default=False,
                        help='Whether or not to add age neuron in abstract risk model')
    parser.add_argument('--time_embed_dim', type=int, default=128, help="Representation layer size for time embeddings.")

    # Learning Hyper-params
    parser.add_argument('--loss_fn', type=str, default="binary_cross_entropy_with_logits",
                        help='loss function to use, available: [Xent (default), MSE]')
    parser.add_argument('--optimizer', type=str, default="adam", help='The optimizer to use during training. '
                                                                      'Choose from [default: adam, adagrad, sgd]')
    parser.add_argument('--train_batch_size', type=int, default=64, help="Batch size used when training the model.")
    parser.add_argument('--eval_batch_size', type=int, default=16,
                        help="Batch size used when evaluating the model. Note that evaluation step takes all valid "
                             "partial trajectories from each patient, therefore would consume higher memory per batch. "
                             "One can adjust this accordingly using this option. ")
    parser.add_argument('--max_batches_per_train_epoch', type=int, default=10000,
                        help='max batches to per train epoch. [default: 10000]')
    parser.add_argument('--max_batches_per_dev_epoch', type=int, default=10000,
                        help='max batches to per dev epoch. [default: 10000]')
    parser.add_argument('--exhaust_dataloader', action='store_true', default=False,
                        help='Whether to truncate epoch to max batches per dataset epoch or to exhaust the full '
                             'dataloader. Useful when the whole data is too large. Default: False.')
    parser.add_argument('--init_lr', type=float, default=0.001, help='The initial learning rate [default: 0.001]')
    parser.add_argument('--lr_decay', type=float, default=1., help='Decay of learning rate [default: no decay (1.)]')
    parser.add_argument('--momentum', type=float, default=0, help='Momentum to use with SGD')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 Regularization penaty [default: 0]')
    parser.add_argument('--patience', type=int, default=5,
                        help='Number of epochs without improvement on dev before halving learning rate or early '
                             'stopping. [default: 5]')
    parser.add_argument('--tuning_metric', type=str, default='36month_auroc',
                        help='Metric to judge dev set results. Possible options include auc, loss, accuracy and etc.')
    parser.add_argument('--epochs', type=int, default=20, help='Total number of epochs for training [default: 20].')

    # evaluation
    parser.add_argument('--eval_auroc', action='store_true', default=False, help='Whether to calculate AUROC')
    parser.add_argument('--eval_auprc', action='store_true', default=False, help='Whether to calculate AUPRC')
    parser.add_argument('--eval_mcc', action='store_true', default=False, help='Whether to calculate MCC')
    parser.add_argument('--eval_c_index', action='store_true', default=False, help='Whether to calculate c-Index')

    # Where to store stuff
    parser.add_argument('--save_dir', type=str, required=True, help='The output file location.')
    parser.add_argument('--model_dir', type=str, default="snapshots", help='The path to the library of trained models.')
    parser.add_argument('--exp_id', type=str, default='debug', help='The identifier/name for each run')
    parser.add_argument('--time_logger_verbose', type=int, default=2,
                        help='Verbose of logging (1: each main, 2: each epoch, 3: each step). Default: 2.')
    parser.add_argument('--time_logger_step', type=int, default=1,
                        help='Log the time elapse every how many iterations - 0 for no logging.')
    parser.add_argument('--resume_from_result', type=str, default=None,
                        help='The full path to the `*.result` file to reload. Note that all `*.result*` files are '
                             'required to be in the same folder and that the corresponding checkpoint files are '
                             'required in the model library folder.')

    args = parser.parse_args() if args_str is None else parser.parse_args(args_str.split())

    # Generate a few more flags to help model construction
    args.results_path = os.path.join(args.save_dir, args.exp_id) + ".results"
    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = 'cuda' if args.cuda else 'cpu'
    args.num_years = max(args.month_endpoints) / 12

    # Check whether the current args is legal.
    if args.train:
        assert args.dev, Exception("[E] --dev is disabled. The dev dataset is required if --train for turning purpose.")
    
    if args.test or args.dev:
        if not any([True for argument, values in args.__dict__.items() for metric in argument.split('_')[-1:]
                    if metric in args.tuning_metric and values]) and args.tuning_metric is not 'loss':
            raise Exception("[E] Tuning metric {} is not computed in Eval metric! Aborting.".format(args.tuning_metric))

        assert any([args.eval_auroc, args.eval_auprc, args.eval_mcc, args.eval_c_index]),\
            Exception("[E] At least one evaluation metric needs to be enabled. "
                    "Choose one or more from AUPRC, AUROC, MCC or c-Index")

    if args.num_heads is not None and args.model_name != 'transformer':
        warnings.warn("[W] The `num_heads` is intended to work with transformer only. "
                      "Setting this for `{}` will have no effects.".format(args.model_name))
    elif args.num_heads is None and args.model_name == 'transformer' and not args.resume_from_result:
        raise Exception("[E] The `num_heads` is required for transformer models. Add that to your config and try again.")

    # Set up initial state for learning rate
    args.optimizer_state = None
    args.current_epoch = None
    args.lr = None
    args.epoch_stats = None
    args.step_indx = 1

    # Resume experiments
    if args.resume_from_result:
        resumed_args = pickle.load(open(args.resume_from_result, "rb"))
        overwrite_args = ['--exclusion_interval', '--metadata_path', 
                        '--dataset', '--save_dir','--num_workers', 
                        '--eval_batch_size', '--max_batches_per_dev_epoch', 
                        '--resume_from_result']
        keep_args_from_config = ['train', 'dev', 'test', 'attribute']
        for a in overwrite_args:
            if a in sys.argv:  # if specified differently for continued experiments
                keep_args_from_config.append(a.replace('--',''))
        for k, v in resumed_args.items():
            if k not in keep_args_from_config:
                args.__dict__[k] = v

        resumed_args = Dict2Args(resumed_args)
        args.snapshot = state.get_model_path(resumed_args)
        args.device = args.device if torch.cuda.is_available() else 'cpu'

    return args


def parse_dispatcher_config(config):
    """
        Parses an experiment config, and creates jobs. For flags that are expected to be a single item,
        but the config contains a list, this will return one job for each item in the list.

        Args:
            config - experiment_config json file
        Returns:
            jobs - a list of flag strings, each of which encapsulates one job.
            * Example: --train --cuda --dropout=0.1 ...

    """
    jobs = [""]
    hyperparameter_space = config['search_space']
    hyperparameter_space_flags = hyperparameter_space.keys()
    hyperparameter_space_flags = sorted(hyperparameter_space_flags)
    for ind, flag in enumerate(hyperparameter_space_flags):
        possible_values = hyperparameter_space[flag]

        children = []
        if len(possible_values) == 0 or type(possible_values) is not list:
            raise Exception(POSS_VAL_NOT_LIST.format(flag, possible_values))
        for value in possible_values:
            for parent_job in jobs:
                if type(value) is bool:
                    if value:
                        new_job_str = "{} --{}".format(parent_job, flag)
                    else:
                        new_job_str = parent_job
                elif type(value) is list:
                    val_list_str = " ".join([str(v) for v in value])
                    new_job_str = "{} --{} {}".format(parent_job, flag,
                                                      val_list_str)
                else:
                    new_job_str = "{} --{} {}".format(parent_job, flag, value)
                children.append(new_job_str)
        jobs = children

    return jobs


class Dict2Args(object):
    """
        A helper class for easier attribution retrieval for dict.
    """
    def __init__(self, d):
        self.__dict__ = d


class Yaml2Args(Dict2Args):
    """
        A helper class for easier attribution retrieval for an YAML input.
    """
    def __init__(self, d):
        for item in d:
            if len(d[item]) == 1:
                d[item] = d[item][0]

        super(Yaml2Args, self).__init__(d)


def load_data_settings(args):
    SETTINGS = Yaml2Args(yaml.safe_load(open(args.data_setting_path, 'r')))
    SETTINGS.chapterColors = [(r/255, g/255, b/255) for r, g, b, in SETTINGS.chapterColors]

    CODEDF = pd.read_csv(
        SETTINGS.ICD10_MAPPER_NAME, sep='\t', header=None,
        names=['code', 'description', 'chapter', 'chapter_name', 'block_name', 'block']
    )
    CODEDF = CODEDF.append(pd.read_csv(
        SETTINGS.ICD8_MAPPER_NAME, sep='\t', header=None,
        names=['code', 'description', 'chapter', 'chapter_name', 'block_name', 'block']
    ))
    CODEDF_9 = pd.read_csv(
        SETTINGS.ICD9_MAPPER_NAME, sep='\t', header=0, names=['code_long', 'description', 'shorter description', 'NA']
    )
    CODEDF_9['code'] = [c[:3] for c in CODEDF_9['code_long']]
    CODEDF_9.groupby('code').first()
    CODEDF_w_9 = CODEDF.append(CODEDF_9)
    CODE2DESCRIPTION = (dict(zip(CODEDF_w_9.code, CODEDF_w_9.description)))
    return {'CODE2DESCRIPTION': CODE2DESCRIPTION, 'SETTINGS': SETTINGS, 'CODEDF_w_9': CODEDF_w_9, 'CODEDF': CODEDF}


def md5(key):
    """
        returns a hashed with md5 string of the key
    """
    return hashlib.md5(key.encode()).hexdigest()


def get_code(args, event, char=False):
    if type(event) is dict:
        code = event['codes']
    else:
        code = event

    if char:
        trunc_level = max(args.icd8_level, args.icd10_level) + 1
        return '-' * (trunc_level - len(code)) + code[:trunc_level]
    code = code.replace('.', '')
    if code[0] == 'D' and not code[1].isdigit():  # this means it is a SKS code
        return code[:args.icd10_level + 1]  # TODO data - check replacement before truncation or after?
    elif code.isdigit():
        return code[:args.icd8_level]
    elif code[0] == 'Y' or code[0] == 'E':  # TODO data - separate SKS or RPDR code by the data class not by filtering
        return code[:args.icd8_level + 1]
    else:
        return code[:args.icd10_level]
