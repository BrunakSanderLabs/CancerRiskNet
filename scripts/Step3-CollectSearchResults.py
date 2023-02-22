# This scripts is used to collect and parse the available results of a single search, the performance is summarized in a
# summary file.
# The script also saves a plot for each best experiment of at each exclusion interval (TODO: still valid?).
import argparse
import os
import csv
import json
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pickle as pkl
from itertools import combinations
from os.path import dirname, realpath
sys.path.insert(0, dirname(dirname(realpath(__file__))))
import cancerrisknet.utils.parsing as parsing

matplotlib.rcParams.update({'figure.max_open_warning': 0})
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

CONFIG_NOT_FOUND_MSG = "[Step3-CollectSearchResults][1/3][INFO] {}: {} config file not specified. Summarizing " \
                       "from all available result files instead."
SUMMARIZING_MSG = "[Step3-CollectSearchResults][1/3][INFO] Summarizing results {} files into {}."
SUMMARYFOUND_MSG = "[Step3-CollectSearchResults][1/3][WARNING] Existing summary file."
SUCESSFUL_SEARCH_STR = "[Step3-CollectSearchResults][2/3][INFO] SUCCESS! All summary files are retrieved into {}."
NEXT_COMMAND = "\n[Step3-CollectSearchResults][NEXT] Run the following command to generate the performance table " \
               "(metrics and CI):\n`python scripts/Step5-ResultBootstrap.py --search_metadata {}`" \
               "\nNote that if --test was not run in the original search, " \
               "a resumed run might be needed (see the optional Step 4)."
EPSILON = 1e-9

parser = argparse.ArgumentParser(description='Grid Search Results Collector.')
parser.add_argument("--experiment_config_path", required=True, type=str, help="Path to the search config.")
parser.add_argument('--result_dir', type=str, default="results", help="Where to store logs and result files.")
parser.add_argument('--search_dir', type=str, default="searches", help="Where to store the information for the search.")
parser.add_argument('--metric', type=str, default="dev_{}month_auprc", help='Metric to use for ordering the results.')
parser.add_argument('--overwrite', action='store_true', default=False, help='Overwrite existing summary file.')
parser.add_argument('--skip_loading', action='store_true', default=False)
args = parser.parse_args()


def update_summary_with_results(result_path, log_path, summary, summary_path):
    assert result_path is not None
    print('[Step3-CollectSearchResults][2/3][INFO] Updating summary with result {}'.format(result_path))

    try:
        results = pkl.load(open(result_path, 'rb'))
    except FileNotFoundError:
        print("[Step3-CollectSearchResults][2/3][ERR] Experiment failed or the result file is in another location! "
              "Logs are located at: {}".format(log_path))
        return summary, None

    timepoints = results['month_endpoints']
    metrics = []
    for metric in ['auroc', 'auprc', 'mcc', 'c_index']:
        if results['eval_{}'.format(metric)]:
            metrics.append(metric)

    result_keys = []
    RESULT_KEY_STEMS = ['{}_loss', '{}_c_index', '{}_c_index']

    for i1 in timepoints:
        for i2 in metrics:
            RESULT_KEY_STEMS += ['{}_' + '{}month_{}'.format(i1, i2)]
    LOG_KEYS = ['result_path', 'model_path', 'log_path']
    for mode in ['epoch_train', 'epoch_dev', 'train', 'dev', 'test']:
        result_keys.extend([k.format(mode) for k in RESULT_KEY_STEMS])
    result_keys = list(set(result_keys))

    try:
        result_dict = {}
        try:
            dict_stats = pkl.load(open('{}.epoch_stats'.format(result_path), 'rb'))
            best_epoch_indx = dict_stats['best_epoch']
            for i in dict_stats:
                if i == 'best_epoch':
                    result_dict.update({i: dict_stats[i]})
                else:
                    result_dict.update({'epoch_' + i: dict_stats[i]})
        except FileNotFoundError:
            pass
        for i in ['dev', 'test']:
            if os.path.exists('{}.{}_stats'.format(result_path, i)):
                dict_stats = pkl.load(open('{}.{}_stats'.format(result_path, i), 'rb'))
                for k, v in dict_stats.items():
                    if k not in result_dict:
                        result_dict[k] = v
            else:
                print("[Step3-CollectSearchResults][2/3][WARNING] Missing {} stats.".format(i))
    except FileNotFoundError:
        print("[Step3-CollectSearchResults][2/3][ERROR] Experiment failed or the result file is in another location!"
              " Logs are located at: {}".format(log_path))
        return summary, None

    present_result_keys = []
    for k in result_keys:
        if k in result_dict:
            if 'epoch' not in k:
                result_dict[k] = result_dict[k][0]
                present_result_keys.append(k)
            else:
                result_dict[k] = result_dict[k][best_epoch_indx]
                present_result_keys.append(k)

    result_dict['log_path'] = log_path
    summary_columns = present_result_keys + LOG_KEYS
    for prev_summary in summary:
        if len(set(prev_summary.keys()).union(set(summary_columns))) > len(summary_columns):
            summary_columns = list(set(prev_summary.keys()).union(set(summary_columns)))

    # Only export keys we want to see in sheet to csv
    summary_dict = {}
    for key in summary_columns:
        if key in result_dict:
            summary_dict[key] = result_dict[key]
        else:
            summary_dict[key] = 'NA'
    summary.append(summary_dict)

    with open(summary_path, 'w') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=summary_columns)
        writer.writeheader()
        for experiment in summary:
            writer.writerow(experiment)
    return summary, results


def vis_df(df, filename, vis_label='untitled', keys=[]):
    if 'Hyperparameter' not in filename:
        keys = [k for k in keys if k in df.keys()]

    with open(filename, 'a') as w:
        w.write('\n' + '#' * 40 + '  ' + vis_label + '  ' + '#' * 40 + '\n\n')

    try:
        out_df = df.loc[:, keys]
        with open(filename, 'a') as w:
            w.write(out_df.to_string())
    except Exception:
        with open(filename, 'a') as w:
            w.write('   '.join(keys))
            w.write('\nN.A.')


def rule_for_best(df, metric, timepoints):
    subset_column_metric = [metric.format(m) for m in timepoints]
    df["score"] = df[subset_column_metric].mean(axis=1)
    df = df.sort_values(by='score', ascending=False)
    return df


def print_grid_performances(summary_df, args_df_subset):
    best_exp_summary = []
    args_df_subset = args_df_subset.fillna(False)
    if len(np.unique(args_df_subset.month_endpoints)) != 1:
        raise NotImplementedError(
            "Cannot estimated yet the best performances when collecting experiments at different points")
    timepoints = args_df_subset.month_endpoints.iloc[0]
    summary_df = rule_for_best(summary_df, args.metric, timepoints)

    exclusion_interval = ['all']
    exclusion_indexes = [args_df_subset.iloc[:, 0].notnull()]

    if 'exclusion_interval' in args_df_subset.columns:
        unique_exclusions = args_df_subset.exclusion_interval.unique().tolist()
        exclusion_interval.extend(unique_exclusions)
        exclusion_indexes.extend([args_df_subset.exclusion_interval == ei for ei in unique_exclusions])

    for einterval, eindex in zip(exclusion_interval, exclusion_indexes):
        args_df_exclusion = args_df_subset[eindex]
        summary_w_all_args = summary_df[['score']].merge(args_df_exclusion, left_index=True,
                                                         right_index=True).sort_values(by='score', ascending=False)

        for use_known_risk, summary_by_risk_factor in summary_w_all_args.groupby('use_known_risk_factors_only'):
            for model, summary_by_model in summary_by_risk_factor.groupby('model_name'):
                if type(einterval) is not int or (type(einterval) is str and not einterval.isdigit()):
                    continue
                summary_by_model = summary_by_model.sort_values(by='score', ascending=False)
                param_for_best_exp = summary_by_model.iloc[0].to_dict()
                if use_known_risk:
                    model += " known risk"
                record = (einterval, model,
                          param_for_best_exp['exp_id'],
                          param_for_best_exp['save_dir'],)
                best_exp_summary.append(record)

        args_df_exclusion = args_df_exclusion.loc[:, args_df_exclusion.astype('str').nunique() > 1]
        summary_w_changing_args = summary_df[['score']].merge(args_df_exclusion, left_index=True,
                                                              right_index=True).sort_values(by='score', ascending=False)

        t_args = [c for c in summary_w_changing_args.columns if c != 'score']
        t_args = [arg for arg in t_args if arg not in ["exp_id", "model_name", "results_path", "save_dir"]]
        combos = [list(combinations(t_args, i)) for i in range(1, min(len(t_args), 3))]

        for aggregate, combo in enumerate(combos):
            fig = plt.figure()
            leftmargin = 0.4
            rightmargin = 0.2
            categorysize = 0.2

            figwidth = leftmargin + rightmargin + ((len(combo[0][0]) + 1) ** 2) * categorysize
            fig.set_size_inches(figwidth, 4 * len(combo))
            idx = 1
            for cols in combo:
                ax = fig.add_subplot(len(combo), 1, idx)
                if len(cols) == 1:
                    param = cols[0]
                else:
                    param = "-".join(cols)
                    summary_w_changing_args[param] = summary_w_changing_args.loc[:, cols].astype(str).agg('-'.join,
                                                                                                          axis=1)
                sns.swarmplot(x=param, y='score', ax=ax, data=summary_w_changing_args.sort_values(by=param))
                # add line to connect similar experiments
                unique_vals = sorted(summary_w_changing_args[param].unique())
                grouped_duplicates = summary_w_changing_args[[c for c in t_args if c not in cols]]
                grouped_duplicates = (grouped_duplicates.groupby(grouped_duplicates.columns.tolist()).apply(
                    lambda x: tuple(x.index)).reset_index(name='idx'))
                paired = [pair for pair in grouped_duplicates.idx.tolist() if len(pair) > 1]
                for pair in paired:
                    scores = [None for _ in range(len(unique_vals))]
                    for exp_id in pair:
                        current_param = summary_w_changing_args.loc[exp_id, param]
                        scores[unique_vals.index(current_param)] = summary_w_changing_args.loc[exp_id, 'score']
                        ax.plot(unique_vals, scores, color='black', alpha=0.2)

                ax.set_ylim(0, 1)
                ax.grid(which='both')
                ax.grid(which='minor', alpha=0.05, linestyle='--')

                idx += 1
            plt.savefig(figure_path + "/grid.aggregate{}_exclusion_{}.png".format(aggregate, einterval),
                        bbox_inches='tight')

    best_exp_summary = pd.DataFrame.from_records(best_exp_summary,
                                                 columns=['exclusion_interval', "model_name", "exp_id", "save_dir"])
    print(best_exp_summary)
    table_path = os.path.join(args.search_dir, "performance_table.csv")
    best_exp_summary.to_csv(table_path, mode='a', index=False)


if __name__ == "__main__":

    print("[Step3-CollectSearchResults][1/3]Start to collect grid exprs...")

    assert os.path.exists(args.experiment_config_path)
    experiment_config_json = json.load(open(args.experiment_config_path, 'r'))
    job_list = parsing.parse_dispatcher_config(experiment_config_json)
    job_ids = [parsing.md5(job) for job in job_list]
    master_id = parsing.md5(''.join(job_list))
    print(SUMMARIZING_MSG.format(len(job_list), args.search_dir + '/master.{}.summary'.format(master_id)))

    with open(args.search_dir + '/master.{}.joblist'.format(master_id), 'w') as out_file:
        out_file.write("{}\n".format(args.experiment_config_path))
        for job in job_list:
            if '--' in job:
                job_md5 = parsing.md5(job)
            else:
                job_md5 = job
            out_file.write("{}, {}\n".format(job_md5, job))

    figure_path = os.path.join(args.search_dir, 'collection_summary')
    os.makedirs(figure_path, exist_ok=True)
    summary_path = figure_path + '/master.{}.summary'.format(master_id)
    summary = []
    args_dict = {}

    if not args.skip_loading:  # TODO: why do we want to skip loading?
        print("[Step3-CollectSearchResults][2/3] Start to load *.results files...")
        for job in job_ids:
            result_path = os.path.join(args.result_dir, job + '.results')
            log_path = os.path.join(args.result_dir, job + '.txt')
            summary, job_args = update_summary_with_results(result_path, log_path, summary, summary_path)
            if not job_args:
                continue
            sorted_key_args = sorted(job_args.keys())
            args_dict.update({job: [job_args[k] for k in sorted_key_args]})
        print(SUCESSFUL_SEARCH_STR.format(summary_path))

    args_df = pd.DataFrame.from_dict(args_dict, orient='index', columns=sorted_key_args)
    print("[Step3-CollectSearchResults][3/3] Start exporting... ")
    exp = master_id
    summary_df = pd.read_csv(summary_path)
    summary_df.index = [os.path.basename(log_path).split('.')[0] for log_path in summary_df.log_path]
    configs = pd.read_csv(args.search_dir + '/master.' + exp + '.joblist')

    summary_txt = os.path.join(figure_path, 'master.summary.{}.txt'.format(exp))
    if os.path.exists(summary_txt):
        if args.overwrite:
            print('Overwriting...')
            os.remove(summary_txt)
        else:
            print("Aborting to avoid overwriting...")
            sys.exit(1)

    args_vis = args_df.loc[:, (args_df != args_df.iloc[0]).any()]
    summary_args = os.path.join(figure_path, 'master.args_vis.{}.csv'.format(exp))
    args_df.to_csv(summary_args)
    summary_csv = os.path.join(figure_path, 'master.summary_df.{}.csv'.format(exp))
    summary_df.to_csv(summary_csv)

    summary_df_test_only = summary_df.loc[:, [k for k in summary_df.keys() if 'test' in k and 'roc' in k]]
    merge_df = args_df.merge(summary_df_test_only, left_index=True, right_index=True)
    merge_df_brief = args_vis.merge(summary_df_test_only, left_index=True, right_index=True)
    merge_df.to_csv(os.path.join(figure_path, 'master.merge_info.{}.csv'.format(exp)))
    merge_df_brief.to_csv(os.path.join(figure_path, 'master.merge_info_brief.{}.csv'.format(exp)))

    print_grid_performances(summary_df, args_df)

    partitions = ['epoch_train', 'epoch_dev', 'train', 'dev', 'test']
    time_series = ['3month', '6month', '12month', '36month', '60month', '120month']
    vis_df(args_df, summary_txt, 'Hyperparameter summary', (args_df != args_df.iloc[0]).any())
    vis_df(summary_df, summary_txt, 'C-Index (all 3 datasets)', [p + '_c_index' for p in partitions])
    vis_df(summary_df, summary_txt, 'AUROC - (Dev) Full eval', ['dev_{}_auroc'.format(s) for s in time_series])
    vis_df(summary_df, summary_txt, 'AUPRC - (Dev) Full eval', ['dev_{}_auprc'.format(s) for s in time_series])

    print(NEXT_COMMAND.format(os.path.join(args.search_dir, "performance_table.csv"))) #TODO performance_table.csv name could also be an argument in parser
