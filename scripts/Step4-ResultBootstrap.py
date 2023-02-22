#
# Starting froma config file generates all the metrics and figures coordinates for the experiment to collect.
# It generates the Tables in supplementary and the main table used to generate the plot_model_performance.
# 
import argparse
import os
import pickle
import csv
import json
import sys
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import sklearn.metrics
from matplotlib.gridspec import GridSpec
from sklearn.metrics._ranking import _binary_clf_curve
from tqdm import tqdm
import multiprocessing
from multiprocessing import get_context
import matplotlib.pyplot as plt
from os.path import dirname, realpath
sys.path.insert(0, dirname(dirname(realpath(__file__))))
from cancerrisknet.utils.eval import include_exam_and_determine_label, get_probs_golds


def mean_confidence_interval(data, *args, confidence=0.95):
    a = 1.0*np.array(data)
    k = 1 - confidence
    k *= 100
    hm, m, mh = np.percentile(a, (k/2, 50, 100-k/2))
    res = hm, m, mh, *args
    return res


def get_boot_metric_clf(n):

    if n > 0:
        sample = np.random.choice(probs_for_eval.size, probs_for_eval.size, replace=True)
        probs = probs_for_eval[sample]
        golds = golds_for_eval[sample]
    else:
        probs = probs_for_eval
        golds = golds_for_eval
    
    fps, tps, thresholds = _binary_clf_curve(
        golds, probs, pos_label=1)

    if len(thresholds) == 1:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    p_count = tps[-1]
    n_count = fps[-1]

    fns = p_count - tps
    tns = n_count - fps
    precisions = tps / (tps + fps)
    precisions[np.isnan(precisions)] = 0
    recalls = tps / p_count

    fprs = fps / n_count
    tprs = tps / p_count

    np.seterr(invalid='ignore')
    with np.errstate(divide='ignore'):
        odds_ratio = np.nan_to_num((tps / fps) / np.nan_to_num(fns / tns), posinf=0, nan=0)
    ps = tps + fps
    f1s = 2 * tps / (ps + p_count)
    incidence_ = np.round(p_count / (p_count + n_count), 4)

    auprc_ = sklearn.metrics.auc(recalls, precisions)
    auroc_ = sklearn.metrics.auc(fprs, tprs)

    idx_prc = np.nanargmax(f1s)
    precision_ = precisions[idx_prc]
    recall_ = recalls[idx_prc]
    odds_ratio_ = odds_ratio[idx_prc]
    tpr_ = tprs[idx_prc]
    fpr_ = fprs[idx_prc]
    threshold_ = thresholds[idx_prc]

    tn, fp, fn, tp = tns[idx_prc], fps[idx_prc], fns[idx_prc], tps[idx_prc]

    if n == 0:
        return {"precision":precisions[::20].tolist(), 
                "recall":recalls[::20].tolist(), 
                "tpr": tprs[::20].tolist(), 
                "fpr": fprs[::20].tolist(), 
                "odds_ratio": odds_ratio[::20].tolist(), 
                "thresholds":thresholds[::20].tolist(), 
                "cm":[tn, fp, fn, tp]}
    else:
        return auroc_, fpr_, tpr_, auprc_, precision_, recall_, odds_ratio_, incidence_, threshold_


def child_initialize(_probs_for_eval, _golds_for_eval):
    global probs_for_eval, golds_for_eval
    probs_for_eval = _probs_for_eval
    golds_for_eval = _golds_for_eval


def get_performance_ci(probs_for_eval, golds_for_eval, model_name, prediction_interval, exclusion_interval, exp_id, n_boot=2):

    with get_context("spawn").Pool(min(200, n_boot), initializer=child_initialize, initargs=(probs_for_eval, golds_for_eval)) as pool:
        metrics = pool.map(get_boot_metric_clf, range(n_boot))
    # metrics = [get_boot_metric(n) for n in tqdm(range(n_boot))]
    curves = (None, json.dumps(metrics.pop(0)), None, model_name, 'curves', prediction_interval, exclusion_interval, exp_id)
    auroc, fpr, tpr, auprc, precision, recall, odds_ratio, incidence, threshold = zip(*metrics)
    incidence_ci = mean_confidence_interval(incidence, model_name, 'incidence', prediction_interval, exclusion_interval, exp_id)
    auroc_ci = mean_confidence_interval(auroc, model_name, 'auroc', prediction_interval, exclusion_interval, exp_id)
    fpr_ci = mean_confidence_interval(fpr, model_name, 'fpr', prediction_interval, exclusion_interval, exp_id)
    tpr_ci = mean_confidence_interval(tpr, model_name, 'tpr', prediction_interval, exclusion_interval, exp_id)
    auprc_ci = mean_confidence_interval(auprc, model_name, 'auprc', prediction_interval, exclusion_interval, exp_id)
    precision_ci = mean_confidence_interval(precision, model_name, 'precision', prediction_interval, exclusion_interval, exp_id)
    recall_ci =  mean_confidence_interval(recall, model_name, 'recall', prediction_interval, exclusion_interval, exp_id)
    odds_ratio_ci =  mean_confidence_interval(odds_ratio, model_name, 'odds_ratio', prediction_interval, exclusion_interval, exp_id)
    threshold_ci =  mean_confidence_interval(threshold, model_name, 'thershold', prediction_interval, exclusion_interval, exp_id)
    return curves, incidence_ci, auroc_ci, fpr_ci, tpr_ci, auprc_ci, precision_ci, recall_ci, odds_ratio_ci, threshold_ci


def get_slice(df, model_name=None, metric_name=None, prediction_interval=None, exclusion_interval=None):
    if model_name is not None:
        df = df.loc[df.Model == model_name]
    if metric_name is not None:
        if type(metric_name) is str:
            df = df.loc[df.Metric == metric_name]
        else:
            df = df.loc[[i in metric_name for i in df.Metric]]
    if prediction_interval is not None:
        df = df.loc[df['Prediction Interval'] == prediction_interval]
    if exclusion_interval is not None:
        df = df.loc[df['Exclusion Interval'] == exclusion_interval]
    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Grid Search Results Collector.')
    parser.add_argument("--search_metadata", required=True, type=str, help="Path of the config for the experiment to use to generate the table")
    parser.add_argument('--bootstrap_size', type=int, default=200, help='Number of bootstraps')
    parser.add_argument('--n_samples', type=int, default=1, help='Downsample the output prediction. 1 for every N samples.')
    parser.add_argument('--filename', type=str, default='bootstrap', help='Downsample the output prediction. 1 for every N samples.')

    args = parser.parse_args()
    
    assert args.bootstrap_size > 1, "Choose a boot size higher than 1. "
    if 'json' in args.search_metadata:
        best_exp_ids_config = json.load(open(args.search_metadata, 'r'))
    else:
        best_exp_ids_config = pd.read_csv(args.search_metadata)
        best_exp_ids_config = best_exp_ids_config.drop_duplicates()
        best_exp_ids_config = best_exp_ids_config.to_dict('list')
    prefix = '{}_TableS4'.format(args.filename)
    
    metrics_records = []
    for i, (exp_id, save_dir, model_name, exclusion_interval) in enumerate(zip(best_exp_ids_config['exp_id'], best_exp_ids_config['save_dir'], best_exp_ids_config['model_name'], best_exp_ids_config['exclusion_interval'])):
        printing_prefix = "[Step5-ResultsBootstrap][{}/{}][{}]".format(i+1, len(best_exp_ids_config['exp_id']), exp_id)
        results_path = os.path.join(save_dir, "{}.results".format(exp_id))
        test_preds_path = os.path.join(save_dir, "{}.results.test_preds".format(exp_id))
        if (not os.path.exists(results_path)) or (not os.path.exists(test_preds_path)):
            print(printing_prefix, "[WARNING] File not found for {} at {} ! "
                                   "Consider not to skip step 4.".format(exp_id, test_preds_path))
            continue
        results = pickle.load(open(results_path, 'rb'))
        test_preds = pickle.load(open(test_preds_path, 'rb'))
        print(printing_prefix, "[INFO] Data loaded from {}... ".format(test_preds_path))
        for index, month in enumerate(results['month_endpoints']):
            probs_for_eval, golds_for_eval = get_probs_golds(test_preds, index=index)
            probs_for_eval = np.array(probs_for_eval)[::args.n_samples]
            golds_for_eval = np.array(golds_for_eval)[::args.n_samples]
            
            if not np.sum(golds_for_eval) > 0:
                continue

            print(printing_prefix, "Processing time interval: {} [{}/{}].".format(
                month, index+1, len(results['month_endpoints'])))
            experiment_performance = get_performance_ci(probs_for_eval, golds_for_eval, model_name, month,
                                                        exclusion_interval, exp_id, n_boot=args.bootstrap_size)
            metrics_records.extend(experiment_performance)

    os.makedirs(os.path.join(os.path.dirname(args.search_metadata), 'figures'), exist_ok=True)
    os.chdir(os.path.join(os.path.dirname(args.search_metadata), 'figures'))
    df = pd.DataFrame.from_records(metrics_records, columns=['ci_low', 'Median', 'ci_high', 'Model', 'Metric', 'Prediction Interval', 'Exclusion Interval', 'exp_id'])
    df = df.astype({'Prediction Interval':'int32', "Exclusion Interval":'int32'})
    df.to_csv(prefix + '.Performance_table.csv', sep=',', index=False)

    df['print_aucs'] = ["{:.3f} ({:.3f}-{:.3f})".format(i['Median'], i['ci_low'], i['ci_high'])
                        if i['Metric'][:2] == 'au' else np.nan for i in df.iloc]
    df['print_specificity'] = ["{:.2%} ({:.2%}-{:.2%})".format(1 - i['Median'], 1 - i['ci_high'], 1 - i['ci_low'])
                               if i['Metric'] == 'fpr' else np.nan for i in df.iloc]
    df['print_others'] = ["{:.1%} ({:.1%}-{:.1%})".format(i['Median'], i['ci_low'], i['ci_high'])
                          if i['Metric'][:2] not in ['au', 'fp'] and i['Metric'] not in ['curves'] else np.nan for i in df.iloc]
    df['print_merged'] = [i['print_specificity'] if i['Metric'] == 'fpr' else i['print_others'] for i in df.iloc]

    get_slice(df, metric_name='auroc').pivot_table(
        values='print_aucs', columns=['Prediction Interval'],
        index=['Model', 'Exclusion Interval'], aggfunc=lambda x: [v for v in x]
    ).fillna("-").to_csv(prefix + ".Performance_summary_auroc.csv")

    get_slice(df, metric_name='auprc').pivot_table(
        values='print_aucs', columns=['Prediction Interval'],
        index=['Model', 'Exclusion Interval'], aggfunc=lambda x: [v for v in x]
    ).fillna("-").to_csv(prefix + ".Performance_summary_auprc.csv")

    get_slice(df, metric_name=['precision', 'recall']).pivot_table(
        values='print_others', columns=['Prediction Interval', 'Metric'],
        index=['Model', 'Exclusion Interval'], aggfunc=lambda x: [v for v in x]
    ).fillna("-").to_csv(prefix + ".Performance_summary_pr.csv")

    get_slice(df, metric_name=['fpr', 'precision', 'recall']).pivot_table(
        values='print_merged', index=['Model', 'Exclusion Interval', 'Metric'],
        columns=['Prediction Interval'], aggfunc=lambda x: [v for v in x][0]
    ).fillna('-').to_csv(prefix + ".Performance_summary_p-r-s.csv")

