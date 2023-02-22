#
# This script takes as prediction and attribution from an experiment and generates Figure 4
# 
import argparse
from collections import defaultdict
import os
import pickle
import csv
import json
import sys
import subprocess
from tqdm import tqdm
from os.path import dirname, realpath
sys.path.insert(0, dirname(dirname(dirname(realpath(__file__)))))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
import matplotlib.pylab as pylab
import seaborn as sns
import sklearn.metrics
from sklearn.metrics._ranking import _binary_clf_curve
from cancerrisknet.utils.parsing import load_data_settings
from cancerrisknet.utils.visualization import save_figure_and_subplots
from cancerrisknet.utils.eval import include_exam_and_determine_label


plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=24)  # fontsize of the figure title


def get_probs_golds(test_preds, month='36', args=args):

    probs_for_eval, golds_for_eval, time_to_event = [], [], []
    for prob_arr, censor_time, gold, days_to_censor in tqdm(zip(test_preds["probs"], test_preds["censor_times"], test_preds["golds"], test_preds['days_to_final_censors'])):
        index = args.total_timepoints.index(month)
        include, label = include_exam_and_determine_label(index, censor_time, gold)
        if include:
            probs_for_eval.append(prob_arr[index])
            golds_for_eval.append(label)
            time_to_event.append(days_to_censor)

    return probs_for_eval, golds_for_eval, time_to_event

def load_preds(log_dir, exp_id, n_samples=1):
    try:
        test_preds_path = os.path.join(log_dir, "{}.results.test_preds".format(exp_id))
        print ("Loading {} ..".format(test_preds_path))
        test_preds = pickle.load(open(test_preds_path, 'rb'))
        test_preds = {k:v[::n_samples] for k,v in test_preds.items()}
    except FileNotFoundError:
        print ("Missing {}".format(test_preds_path))
        test_preds = None 
    try:
        censored_attribution_path = os.path.join(log_dir, "{}.results.test_censored_attribution".format(exp_id))
        print ("Loading {} ..".format(censored_attribution_path))
        censored_attribution = pickle.load(open(censored_attribution_path, 'rb'))
    except FileNotFoundError:
        print ("Missing {}".format(censored_attribution_path))
        censored_attribution = None        
    return test_preds, censored_attribution


def code_fn(x):
    if x in CODE2DESCRIPTION:
        return CODE2DESCRIPTION[x][:35] + '\n' + CODE2DESCRIPTION[x][35:]
    if x[0] == 'D':
        if x[1:] in CODE2DESCRIPTION:
            return CODE2DESCRIPTION[x[1:]][:35] + '\n' + CODE2DESCRIPTION[x[1:]][35:]
    else:
        return x
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Grid Search Results Collector.')
    parser.add_argument("--log_dir", required=True, type=str, help="Path to log folder")
    parser.add_argument("--exp_id", required=True, type=str, help="Path to experiement id")
    parser.add_argument("--n_samples", default=1, type=int, help="Path to experiement id")
    parser.add_argument("--timepoints_of_interest", default='3-6-12-36-60-120', type=str, help="Path to prediction pickle file")
    parser.add_argument("--total_timepoints", default='3-6-12-36-60-120', type=str, help="Path to prediction pickle file")

    parser.add_argument("--top_guess", default=5, type=int, help="How many top disease to look at")
    parser.add_argument('--indipendent_eval', action='store_true', default=False, help='choose between indipendent or cumulative evalution')
    parser.add_argument('--apply_color', action='store_true', default=False, help='choose between indipendent or cumulative evalution')
    parser.add_argument('--suffix', type=str, default='', help='fig suffix')
    parser.add_argument('--data_setting_path', type=str, default='data/settings.yaml',
                        help="Path of yaml with data specific settings")
    args = parser.parse_args()
    
    args.timepoints_of_interest = args.timepoints_of_interest.split('-')
    args.total_timepoints = args.total_timepoints.split('-')

    data_settings = load_data_settings(args)
    CODE2DESCRIPTION = data_settings['CODE2DESCRIPTION']
    CODEDF = data_settings['CODEDF']
    chapt2chaptname = dict(zip(CODEDF.chapter, CODEDF.chapter_name))
    icd2chapter = dict(zip(CODEDF.code, CODEDF.chapter))
    test_preds, censored_attribution = load_preds(args.log_dir, args.exp_id, n_samples=args.n_samples)

    if not test_preds:
        sys.exit()

    time_bins = np.array([6,12,24,36])
    attribution_rank = {k: defaultdict(list) for k in time_bins}
    if censored_attribution:
        for time in sorted(censored_attribution):
            if time > time_bins[-1]:
                continue
            time_idx = time_bins[time_bins>=time][0]
            for code, code_attr in censored_attribution[time].items():
                attribution_rank[time_idx][code].extend(code_attr)
            
        for time in attribution_rank:
            attribution_rank[time] = sorted([(code_fn(k), np.sum(v), k) for k,v in attribution_rank[time].items()], key=lambda x: x[1], reverse=True)
        
        attribution_records = []
        color_records = []
        for top_guess in range(args.top_guess):
            time_records = [attribution_rank[t][top_guess][0] for t in sorted(attribution_rank) if attribution_rank[t]]
            code_for_color = [attribution_rank[t][top_guess][-1] for t in sorted(attribution_rank) if attribution_rank[t]]

            color_records.append([SETTINGS.chapterColors[icd2chapter[el[1:]]-1] if el[1:] \
                in icd2chapter and el[0]=='D' else 'w' for el in code_for_color])
            attribution_records.append(time_records)
    else:
        attribution_records = None

    gs = GridSpec(len(args.timepoints_of_interest), 8)
    fig = plt.figure(figsize=[28, len(args.timepoints_of_interest) * 10])
    ax_objs = []
    gs_r= GridSpec(len(args.timepoints_of_interest), 8)
    fig_r = plt.figure(figsize=[28, len(args.timepoints_of_interest) * 10])
    ax_objs_r = []

    for i, month in enumerate(args.timepoints_of_interest):
        probs_for_eval, golds_for_eval, time_to_event = get_probs_golds(test_preds, month=month, args=args)
        #metrics generation
        fps, tps, thresholds = _binary_clf_curve(
            golds_for_eval, probs_for_eval, pos_label=1)

        p_count = tps[-1]
        n_count = fps[-1]

        fns = p_count - tps
        tns = n_count - fps
        precision = tps / (tps + fps)
        precision[np.isnan(precision)] = 0
        recall = tps / p_count
        with np.errstate(divide='ignore'):
            odds_ratio = np.nan_to_num((tps / fps) / np.nan_to_num(fns / tns), posinf=0, nan=0)
        fpr = fps / n_count
        tpr =  tps/ p_count
        ps = tps + fps
        f1s = 2*tps / (ps + p_count)
        incidence = np.round(p_count / (p_count+n_count), 4)

        threshold_index = np.argmax(f1s)

        tp_trajectories = np.logical_and(probs_for_eval>=thresholds[threshold_index], golds_for_eval)
        fn_trajectories = np.logical_and(probs_for_eval<thresholds[threshold_index], golds_for_eval)

        visualization_records = []
        recall_at_timepoints = []
        bins = np.linspace(0,int(month)*30, 20, dtype=np.int32)
        for tp, fn, time, g in zip(tp_trajectories, fn_trajectories, time_to_event, golds_for_eval):
            if tp:
                visualization_records.append(("True positive", time))
            elif fn:
                visualization_records.append(("False negative", time))
            else:
                assert not g
            if g:
                recall_at_timepoints.append((bins[bins>=time][0], tp))

        visualization_df = pd.DataFrame.from_records(visualization_records, columns=['Prediction', 'Time to event'], )
        ax_objs.append(fig.add_subplot(gs[i:i+1, :2]))
        sns.histplot(data=visualization_df, x='Time to event', hue='Prediction', ax=ax_objs[-1], multiple='stack')
        ax_objs[-1].set(title=f"Prediction at {month} months")

        visualization_df_r = pd.DataFrame.from_records(recall_at_timepoints, columns=['Time to event', 'Recall'], )
        ax_objs_r.append(fig_r.add_subplot(gs_r[i:i+1, :2]))
        
        recall_ = visualization_df_r.groupby('Time to event').mean().Recall.values
        palette_ = sns.color_palette("Greens_d", len(recall_))
        rank = recall_.argsort().argsort()
        ranked_palette = np.array(palette_)[rank]

        sns.barplot(data=visualization_df_r, 
            x='Time to event', y='Recall', ax=ax_objs_r[-1], 
            ci=None, palette=ranked_palette,  edgecolor=".2")
        ax_objs_r[-1].set(title=f"Prediction at {month} months")
        ax_objs_r[-1].locator_params(nbins=10, axis='x')
        
        if args.apply_color:
            cell_colors = color_records
        else:
            cell_colors = None
        if int(month) == 36 and attribution_records:
            top_guess = pd.DataFrame.from_records(attribution_records, 
                columns=[f"Cancer in {m} months" for m in list(sorted(attribution_rank))])
            top_guess.to_csv(f"figures/censored_attribution_top{args.top_guess}.tsv", sep='\t', index=False)
            ax_objs.append(fig.add_subplot(gs_r[i:i+1, 2:]))
            ax_objs_r.append(fig_r.add_subplot(gs_r[i:i+1, 2:]))
            tb = ax_objs[-1].table(cellText=top_guess.values, 
                colLabels=top_guess.columns, 
                loc='center',cellLoc='center',
                rowLabels=(top_guess.index + 1),
                colColours=['gainsboro'] * len(top_guess),
                cellColours=cell_colors,
                bbox = [0.0, 0.0, 1.0, 1.0])
            tb.auto_set_font_size(False)
            tb.set_fontsize(14)
            tb_r = ax_objs_r[-1].table(cellText=top_guess.values, 
                colLabels=top_guess.columns, 
                loc='center',cellLoc='center',
                rowLabels=(top_guess.index + 1),
                colColours=['gainsboro'] * len(top_guess),
                cellColours=cell_colors,
                bbox = [0.0, 0.0, 1.0, 1.0])
            tb_r.auto_set_font_size(False)
            tb_r.set_fontsize(14)
            spines = ["top","right","left","bottom"]
            for s in spines:
                ax_objs[-1].spines[s].set_visible(False)
                ax_objs[-1].axis('off')
                ax_objs_r[-1].spines[s].set_visible(False)
                ax_objs_r[-1].axis('off')
            
            for (row, col), cell in tb.get_celld().items():
                if (row == 0):
                    cell.set_text_props(fontproperties=FontProperties(weight='bold', size=18),)
                cell.set_alpha(.7)
            for (row, col), cell in tb_r.get_celld().items():
                if (row == 0):
                    cell.set_text_props(fontproperties=FontProperties(weight='bold', size=18),)  
                cell.set_alpha(.7)
    fig.suptitle("Time to event Recall and Feature contribution", fontsize=16)
    plt.tight_layout()
    save_figure_and_subplots(f'figures/time_to_event_for_tp_fn{args.suffix}', fig, format='png', dpi=300)
    fig.suptitle("Time to event Recall and Feature contribution", fontsize=16)
    plt.tight_layout()
    save_figure_and_subplots(f'figures/time_to_event_for_recall{args.suffix}', fig_r, format='png', dpi=300)

# python scripts/summarizer/plot_time_from_assessment_to_pc.py \
# --log_dir configs/figures/model_performances.json \
# --exp_id