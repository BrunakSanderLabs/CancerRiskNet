# This scripts is used to check the environment and existence of necessary files. Python version and package
# dependencies are also checked. If no code map is detected for the json data, one would be generated automatically.
import argparse
import os
import json
import sys
from datetime import datetime
import shutil
import stat
import subprocess
import random
import pandas as pd
import warnings
from os.path import dirname, realpath
sys.path.insert(0, dirname(dirname(realpath(__file__))))
import cancerrisknet.utils.parsing as parsing


CONFIG_NOT_FOUND_MSG = "ERROR! {}: {} config file not found."
RESULTS_PATH_APPEAR_ERR = 'ALERT! Existing results for the same config(s).'
SUCESSFUL_SEARCH_STR = "Finished! All worker experiments scheduled!"
NO_SCHEDULER_FOUND = "No scheduler found in registry with name {}. Chose between {}"
COMMAND_TO_COLLECT_SEARCH = "\nGrid search finished successfully. \nRun the following commmand to collect" \
                            " the grid:\n`python -u scripts/Step3-CollectSearchResults.py " \
                            "--experiment_config_path {} --search_dir {} --result_dir {}`."
SCHEDULER_REGISTRY = {}

parser = argparse.ArgumentParser(description='CancerRiskNet Grid Search Scheduler.')
parser.add_argument("--search_name", default="untitled-search", type=str, help="The name of the search.")
parser.add_argument("--experiment_config_path", required=True, type=str, help="Path to the search config.")
parser.add_argument("--n_workers", type=int, default=1, help="How many worker nodes to schedule?")
parser.add_argument('--save_dir', type=str, default="results",
                    help="The location to store logs and job level result files")
parser.add_argument('--search_summary_dir', type=str, default="searches", help="The location to store search summary.")
parser.add_argument('--scheduler', type=str, default="single_node_scheduler",
                    help="Which scheduler to use. Choose from [\n'single_node_scheduler', \n'gcp_scheduler', "
                         "\n'torque_scheduler'],")
parser.add_argument("--machines", type=str, default='configs/avai_machines.txt',
                    help="Only for scheduler gcp: the status sheet of available machines.")
parser.add_argument('--shuffle_experiment_order', action='store_true', default=False,
                    help='Whether to shuffle the order of experiments during grid search.')
args = parser.parse_args()
args.save_dir = os.path.join(args.save_dir, args.search_name)
args.search_summary_dir = os.path.join(args.search_summary_dir, args.search_name)


def RegisterScheduler(scheduler_name):
    def decorator(f):
        SCHEDULER_REGISTRY[scheduler_name] = f
        return f
    return decorator


@RegisterScheduler("single_node_scheduler")
def single_node_scheduler(workers):
    assert len(workers) == 1 and args.n_workers == 1, "n_workers does not equal to one. Cannot use single node worker."
    worker = workers[0]
    flag_string = ' --experiment_config_path={}/{}.subexp --save_dir={} --summary_path={}/{}.summary'.format(
        args.search_summary_dir, worker, args.save_dir, args.search_summary_dir, worker
    )
    shell_cmd = "python scripts/worker.py {}".format(flag_string)
    jobscript = "{}/{}.sh".format(args.search_summary_dir, worker)
    with open(jobscript, 'w') as f:
        f.write(shell_cmd)
    print("Launching a single-node dispatcher for worker: {}".format(worker))
    st = os.stat(jobscript)
    os.chmod(jobscript, st.st_mode | stat.S_IEXEC)
    subprocess.call(jobscript, shell=True)


@RegisterScheduler("gcp_scheduler")
def gcp_scheduler(workers):
    machines = pd.read_csv(args.machines)
    machines = machines.loc[machines['avai']]
    hostnames = machines['hostname'].values
    machines = machines['ip'].values
    print("[INFO] Scheduler initiated on machines: {}.".format(hostnames))
    if args.n_workers != len(machines):
        warnings.warn("[W] The input n_workers ({}) does not match the available machines ({}).".format(
            args.n_workers, hostnames
        ))
    for worker, hostname, machine in zip(workers, hostnames, machines):
        gcp_worker(args, worker, machine)
        print("[INFO] Worker {} is launched on machine {} ({})".format(worker, hostname, machine))


def gcp_worker(args, worker_id, machine):
    config_path = '{}/{}.subexp'.format(args.search_summary_dir, worker_id)
    os.system("ssh {} 'mkdir -p ~/{}'".format(machine, os.path.dirname(config_path)))
    os.system("scp {} byuan@{}:~/{}".format(config_path, machine, config_path))
    os.system("ssh {} '{}'".format(machine, 'cd ~; git checkout add_code_to_index_map; git pull;'))

    cmd = ['/opt/conda/bin/python', 'scripts/worker.py',
           '--experiment-config-path={}'.format(config_path),
           '--save-dir={}'.format(args.save_dir),
           '--summary_path={}/{}.summary'.format(args.search_summary_dir, worker_id)]
    cmd1 = ' '.join(cmd)
    cmd2 = "scp byuan@{}:~/{}/*.txt ~/{}".format(machine, args.save_dir, args.save_dir)
    os.system("nohup ssh {} 'cd ~/ ; {}; {}; sudo shutdown' &".format(machine, cmd1, cmd2))


@RegisterScheduler("torque_scheduler")
def torque_scheduler(workers):
    for worker in workers:
        flag_string = ' --experiment_config_path={}/{}.subexp --save_dir={} --summary_path={}/{}.summary'.format(
            args.search_summary_dir, worker, args.save_dir, args.search_summary_dir, worker
        )

        shell_cmd = ["#!/bin/bash", "#PBS -l nodes=1:ppn=20:gpus=1", "#PBS -l mem=400gb",
                     "#PBS -l walltime=20:00:00:00", "#PBS -N cancerrisknet", "#PBS -e {}/.$PBS_JOBID.err",
                     "#PBS -o {}/.$PBS_JOBID.out", "python scripts/schedulers/dispatcher.py {}"]

        shell_cmd = '\n'.join(shell_cmd).format(args.search_summary_dir, args.search_summary_dir, flag_string)
        jobscript = "{}/{}.moab.sh".format(args.search_summary_dir, worker)
        with open(jobscript, 'w') as f:
            f.write(shell_cmd)

        print("Launching with moab dispatcher for worker: {}".format(worker))
        subprocess.run(['qsub', jobscript], universal_newlines=True)
        return 0


def generate_config_sublist(experiment_config_json):
    job_list = parsing.parse_dispatcher_config(experiment_config_json)

    if args.shuffle_experiment_order:
        random.shuffle(job_list)

    config_sublists = [[] for _ in range(args.n_workers)]
    for k, job in enumerate(job_list):
        config_sublists[k % args.n_workers].append(job)
    workers = [parsing.md5(''.join(sublist)) for sublist in config_sublists]

    return job_list, config_sublists, workers


if __name__ == "__main__":
    """
        Dispatch a grid search to one or more machines by creating sub-config files and launch multiple workers.
    """
    assert args.scheduler in SCHEDULER_REGISTRY, \
        NO_SCHEDULER_FOUND.format(args.scheduler, list(SCHEDULER_REGISTRY.keys()))

    if not os.path.exists(args.experiment_config_path):
        print(CONFIG_NOT_FOUND_MSG.format("master", args.experiment_config_path))
        sys.exit(1)
    experiment_config = json.load(open(args.experiment_config_path, 'r'))

    job_list, config_sublists, worker_ids = generate_config_sublist(experiment_config_json=experiment_config)
    print("Schduling {} dispatchers for {} jobs!".format(len(config_sublists), len(job_list)))
    [print('Sublist {} : {} jobs.'.format(worker_ids[i], len(sublist))) for i, sublist in enumerate(config_sublists)]

    datestr = datetime.now().strftime("%Y%m%d-%H%M")
    grid_md5 = parsing.md5(''.join(job_list))[:8]
    grid_logs_path_regex = os.path.join(os.path.dirname(args.save_dir),
                                        '{}_{}'.format(os.path.basename(args.save_dir), grid_md5))

    args.search_summary_dir = os.path.join(
        os.path.dirname(args.search_summary_dir),
        '{}_{}_{}'.format(args.search_name, grid_md5, datestr)
    )
    args.save_dir = os.path.join(
        os.path.dirname(args.save_dir),
        '{}_{}_{}'.format(args.search_name, grid_md5, datestr)
    )
    os.makedirs(args.search_summary_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    experiment_summary_path = args.search_summary_dir + '/master.{}.exp'.format(parsing.md5(''.join(job_list)))
    if os.path.exists(experiment_summary_path):
        print(RESULTS_PATH_APPEAR_ERR)
        sys.exit(1)
    else:
        with open(experiment_summary_path, 'w') as out_file:
            out_file.write("worker, job_size\n")
            for i, worker in enumerate(worker_ids):
                out_file.write("{}, {}\n".format(worker, len(config_sublists[i])))
    
    shutil.copy2(args.experiment_config_path, args.search_summary_dir)

    for i, worker in enumerate(worker_ids):
        with open(args.search_summary_dir+'/{}.subexp'.format(worker), 'w') as out_file:
            for experiment in config_sublists[i]:
                out_file.write("{}\n".format(experiment))

    SCHEDULER_REGISTRY[args.scheduler](worker_ids)
    print(SUCESSFUL_SEARCH_STR)
    print(COMMAND_TO_COLLECT_SEARCH.format(args.experiment_config_path, args.search_summary_dir, args.save_dir))
    sys.exit(0)
