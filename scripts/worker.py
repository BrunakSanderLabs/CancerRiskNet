import argparse
import subprocess
import os
import multiprocessing
import pickle
import csv
import json
import sys
from os.path import dirname, realpath
import random
sys.path.insert(0, dirname(dirname(realpath(__file__))))
import cancerrisknet.utils.parsing as parsing
import cancerrisknet.models.factory as model_factory
import cancerrisknet.learn.state_keeper as state

CONFIG_NOT_FOUND_MSG = "ERROR! {}: {} config file not found."
SUCESSFUL_SEARCH_STR = "Finished! Sub experiment search results dumped to {}."
RESULT_KEY_STEMS = ['{}_loss', '{}_c_index']
RESULT_KEY_STEMS += ['{}_'+'{}month_auroc'.format(i) for i in [3, 6, 12, 36, 60, 120]]
RESULT_KEY_STEMS += ['{}_'+'{}month_auprc'.format(i) for i in [3, 6, 12, 36, 60, 120]]
LOG_KEYS = ['results_path', 'model_path', 'log_path']
SORT_KEY = 'dev_36month_auroc'


parser = argparse.ArgumentParser(description='Launches a subset of experiments sequentially on one machine.')
parser.add_argument("--experiment_config_path", required=True, type=str, help="Path of sub-experiment config file.")
parser.add_argument('--summary_path', type=str, default="experiments",
                    help="The summary file is updated after each experiment.")
parser.add_argument('--save_dir', type=str, default="results",
                    help="The location to store logs and detailed job level result files")
parser.add_argument('--gpu', default=0, help='which gpu to run sub-experiment.')
args = parser.parse_args()


def launch_experiment(gpu, flag_string):
    """
        Launch an experiment and direct results_paths and results to a unique filepath.
        Alert if something goes wrong.
    """
    os.makedirs(args.save_dir, exist_ok=True)

    exp_id = parsing.md5(flag_string)
    log_stem = os.path.join(args.save_dir, exp_id)
    log_path = '{}.txt'.format(log_stem)
    results_path = "{}.results".format(log_stem)
    experiment_string = "CUDA_VISIBLE_DEVICES={} python -u scripts/main.py {} --save_dir='{}' --exp_id {}".format(
        gpu, flag_string, args.save_dir, exp_id)

    shell_cmd = "{} > {} 2>&1".format(experiment_string, log_path)
    print("Worker launched:`{}`.".format(shell_cmd))
    subprocess.call(shell_cmd, shell=True)
    return results_path, log_path


def queue_worker(gpu, job_queue, done_queue):
    while not job_queue.empty():
        params = job_queue.get()
        if params is None:
            return
        done_queue.put(
            launch_experiment(gpu, params))


def update_summary_with_results(results_path, log_path, result_keys):
    """
        After each experiment finishes, update results into a summary file for easier reference.
    """
    assert results_path is not None
    summary = []
    try:
        result_dict = pickle.load(open(results_path, 'rb'))
    except FileNotFoundError:
        print("Experiment failed or the result file is in another location! Logs are located at: {}".format(log_path))
        return None
    try:
        epoch_stats = pickle.load(open('{}.{}'.format(results_path, 'epoch_stats'), 'rb'))
    except FileNotFoundError:
        print("Epoch stats not found. Logs are located at: {}".format(log_path))
        epoch_stats = {}
    try:
        dev_stats = pickle.load(open('{}.{}'.format(results_path, 'dev_stats'), 'rb'))
    except FileNotFoundError:
        print("Dev stats not found. Logs are located at: {}".format(log_path))
        dev_stats = {}
    try:
        test_stats = pickle.load(open('{}.{}'.format(results_path, 'test_stats'), 'rb'))
    except FileNotFoundError:
        print("Test stats not found. Logs are located at: {}".format(log_path))
        test_stats = {}

    # Get results from best epoch and move to top level of results dict
    result_dict['log_path'] = log_path
    best_epoch_indx = epoch_stats['best_epoch'] if result_dict['train'] else 0
    present_result_keys = []
    for k in result_keys:
        if (k in test_stats and len(test_stats[k]) > 0) or (k in dev_stats and len(dev_stats[k]) > 0) \
                or (result_dict['train'] and k in epoch_stats and len(epoch_stats[k]) > 0):
            present_result_keys.append(k)
            if 'test' in k:
                result_dict[k] = test_stats[k][0]
            elif 'dev' in k:
                result_dict[k] = dev_stats[k][0]
            else:
                result_dict[k] = epoch_stats[k][best_epoch_indx]

    summary_columns = present_result_keys + LOG_KEYS

    # Only export keys we want to see in sheet to csv
    summary_dict = {}
    for key in summary_columns:
        if key in result_dict:
            summary_dict[key] = result_dict[key]
        else:
            summary_dict[key] = 'NA'
    summary.append(summary_dict)
    if SORT_KEY in summary[0]:
        try:
            summary = sorted(summary, key=lambda x: x[SORT_KEY])
        except Exception:
            pass
    result_dir = os.path.dirname(results_path)
    os.makedirs(result_dir, exist_ok=True)

    # Write summary to csv
    file_exists = os.path.isfile(args.summary_path)
    with open(args.summary_path, 'a+') as out_file:
        writer = csv.DictWriter(out_file, fieldnames=summary_columns)
        if not file_exists:
            writer.writeheader()
        for experiment in summary:
            writer.writerow(experiment)
    return summary


if __name__ == "__main__":
    """
        From a subexp config, take a list of flags as input and sequentially process the queue.
    """

    # load list of job configs
    if not os.path.exists(args.experiment_config_path):
        print(CONFIG_NOT_FOUND_MSG.format("experiment", args.experiment_config_path))
        sys.exit(1)
    with open(args.experiment_config_path) as f:
        job_list = f.read().splitlines()

    # main dispatcher function
    job_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()

    for job in job_list:
        job_queue.put(job)
    print("Launching dispatcher for {} jobs!".format(len(job_list)))
    multiprocessing.Process(target=queue_worker, args=(args.gpu, job_queue, done_queue)).start()

    result_keys = []
    for mode in ['train', 'dev', 'test']:
        result_keys.extend([k.format(mode) for k in RESULT_KEY_STEMS])

    for i in range(len(job_list)):
        args.results_path, args.log_path = done_queue.get()
        update_summary_with_results(args.results_path, args.log_path, result_keys)
        dump_result_string = SUCESSFUL_SEARCH_STR.format(args.results_path)
        print("({}/{}) \t {}".format(i+1, len(job_list), dump_result_string))

    sys.exit(0)
