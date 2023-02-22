import json
import pickle
import tqdm
from collections import Counter, defaultdict
import pandas as pd
import os
import sys
from os.path import dirname, realpath
sys.path.insert(0, dirname(dirname(realpath(__file__))))
from cancerrisknet.utils.parsing import md5
from cancerrisknet.utils.parsing import get_code


NO_DATASET_ERR = "Dataset {} not in DATASET_REGISTRY! Available datasets are {}"
DATASET_REGISTRY = {}
NUM_PICKLES = 50
UNK_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'


def RegisterDataset(dataset_name):
    """Registers a dataset. Used as a decorator."""

    def decorator(f):
        DATASET_REGISTRY[dataset_name] = f
        return f

    return decorator


def get_dataset_class(args):
    if args.dataset not in DATASET_REGISTRY:
        raise Exception(
            NO_DATASET_ERR.format(args.dataset, DATASET_REGISTRY.keys()))

    return DATASET_REGISTRY[args.dataset]


def build_code_to_index_map(args):
    """
        Create a mapping dict for each of the categorical token (diagnosis code) occured in the dataset.
        Require input file `data/all_observed_icd.txt` which should be automatically generated during data pre-processing
        following steps under `scripts/metadata/`.
    """
    print("Building code to index map...")
    vocab_path = os.path.join(
        os.path.dirname(args.metadata_path), os.path.basename(args.metadata_path).replace('.json', '-vocab.txt')
    )
    with open(vocab_path, 'r') as f:
        all_codes = f.readlines()
        all_codes = [x.rstrip('\n') for x in all_codes]

    all_observed_codes = [get_code(args, code) for code in all_codes]
    print("Length of all_observed", len(all_observed_codes))
    all_codes_counts = dict(Counter(all_observed_codes))
    print(len(all_codes_counts))
    all_codes = list(all_codes_counts.keys())
    all_codes_p = list(all_codes_counts.values())
    all_codes_p = [i/sum(all_codes_p) for i in all_codes_p]
    code_to_index_map = {code: i+1 for i, code in enumerate(all_codes)}
    code_to_index_map.update({
        PAD_TOKEN: 0, 
        UNK_TOKEN: len(code_to_index_map)+1
        })
    args.code_to_index_map = code_to_index_map
    args.all_codes = all_codes
    args.all_codes_p = all_codes_p


def get_dataset(args):
    """
        Generate torch-compatible dataset instances for training, evaluation or any other analysis.
    """
    # Depending on arg, build dataset
    metadata = json.load(open(args.metadata_path, 'r'))
    dataset_class = get_dataset_class(args)

    train = dataset_class(metadata, args, 'train') if args.train else []
    dev = dataset_class(metadata, args, 'dev') if args.train or args. dev else []
    test = dataset_class(metadata, args, 'test') if args.test else []

    if args.attribute:
        attr = dataset_class(metadata, args, 'test')
        attr.split_group = "attribute"
    else:
        attr = []

    if args.resume_from_result is None and args.train:
        # Build a new code to index map only during training.
        build_code_to_index_map(args)
        json.dump(args.code_to_index_map, open(args.results_path + '.code_map', 'w'))
    else:
        args.code_to_index_map = json.load(open(args.results_path + '.code_map',  'r'))

    args.index_map_length = len(args.code_to_index_map)

    if args.max_events_length is None:
        args.max_events_length = max([len(record['codes']) for record in train.dataset])

    if args.pad_size is None:
        args.pad_size = args.max_events_length

    args.PAD_TOKEN = PAD_TOKEN
    return train, dev, test, attr, args
