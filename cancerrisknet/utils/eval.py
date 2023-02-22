import sklearn.metrics
from cancerrisknet.utils.c_index import concordance_index
import warnings
import numpy as np
from sklearn.metrics._ranking import _binary_clf_curve


def get_probs_golds(test_preds, index=4):
    """
    Get pairs of predictions and labels that passed the data pre-processing criteria.

    Args:
        test_preds:
        index: the position at which the prediction vector (default: [3,6,12,36,60]) is evaluated.

    Returns:
        A pair of lists with the same length, ready for the use of AUROC, AUPRC, and etc.

    """

    probs_for_eval, golds_for_eval = [], []
    for prob_arr, censor_time, gold in zip(test_preds["probs"], test_preds["censor_times"], test_preds["golds"]):
        include, label = include_exam_and_determine_label(index, censor_time, gold)
        if include:
            probs_for_eval.append(prob_arr[index])
            golds_for_eval.append(label)

    return probs_for_eval, golds_for_eval


def compute_eval_metrics(args, loss, golds, patient_golds, probs, exams, pids, dates, censor_times,
                         days_to_final_censors, stats_dict, key_prefix):
    
    stats_dict['{}_loss'.format(key_prefix)].append(loss)
    preds_dict = {
        'golds': golds,
        'probs': probs,
        'patient_golds': patient_golds,
        'exams': exams,
        'pids': pids,
        'dates': dates,
        'censor_times': censor_times,
        'days_to_final_censors': days_to_final_censors
    }

    log_statement = '-- loss: {:.6f}'.format(loss)

    for index, time in enumerate(args.month_endpoints):
        probs_for_eval, golds_for_eval = get_probs_golds(preds_dict, index=index)

        if args.eval_auroc:
            key_name = '{}_{}month_auroc'.format(key_prefix, time)
            auc = compute_auroc(golds_for_eval, probs_for_eval)
            log_statement += " -{}: {} (n={} , c={} )".format(key_name, auc, len(golds_for_eval), sum(golds_for_eval))
            stats_dict[key_name].append(auc)

        if args.eval_auprc:
            key_name = '{}_{}month_auprc'.format(key_prefix, time)
            auc = compute_auprc(golds_for_eval, probs_for_eval)
            log_statement += " -{}: {} (n={} , c={} )".format(key_name, auc, len(golds_for_eval), sum(golds_for_eval))    
            stats_dict[key_name].append(auc)

        if args.eval_mcc:
            key_name = '{}_{}month_mcc'.format(key_prefix, time)
            mcc = compute_mcc(golds_for_eval, probs_for_eval)
            log_statement += " -{}: {} (n={} , c={} )".format(key_name, mcc, len(golds_for_eval), sum(golds_for_eval))
            stats_dict[key_name].append(mcc)

    if args.eval_c_index:
        c_index = compute_c_index(probs, censor_times, golds)
        stats_dict['{}_c_index'.format(key_prefix)].append(c_index)
        log_statement += " -c_index: {}".format(c_index)

    return log_statement, stats_dict, preds_dict

def include_exam_and_determine_label(followup, censor_time, gold, cumulative_prediction_interval=True):
    """
        Determine if a given prediction should be evaluated in this pass.

    Args:
        followup:
        censor_time: the position at which the prediction vector (default: [3,6,12,36,60]) is evaluated.
        gold: the ground truth (whether this trajectory is associated with a cancer dianosis or not.
        cumulative_prediction_interval: One of ['c', 'i'].
                                        If 'c' then evalute for the time interval *up to a given time point*,
                                            e.g. there is (not) a cancer dianosis until the 36 months after
                                                 time of assessment.
                                        If 'i' then evalute for the exact time interval for a given time point,
                                            e.g. there is (not) a cancer dianosis occurrence between 12-36 months after
                                                 time of assessment.
    """
    if cumulative_prediction_interval:
        valid_pos = gold and censor_time <= followup
    else:
        valid_pos = gold and censor_time == followup
    valid_neg = censor_time >= followup
    included, label = (valid_pos or valid_neg), valid_pos
    return included, label


def compute_c_index(probs, censor_times, golds):
    try:
        c_index = concordance_index(censor_times, probs, golds)
    except Exception as e:
        warnings.warn("Failed to calculate C-index because {}".format(e))
        c_index = 'NA'
    return c_index


def compute_auroc(golds_for_eval, probs_for_eval):
    try:
        fpr, tpr, _ = sklearn.metrics.roc_curve(golds_for_eval, probs_for_eval, pos_label=1)
        auc = sklearn.metrics.roc_auc_score(golds_for_eval, probs_for_eval, average='samples')
    except Exception as e:
        warnings.warn("Failed to calculate AUROC because {}".format(e))
        auc = 'NA'
    return auc


def compute_auprc(golds_for_eval, probs_for_eval):
    try:
        precisions, recalls, _ = sklearn.metrics.precision_recall_curve(golds_for_eval, probs_for_eval, pos_label=1)
        auc = sklearn.metrics.auc(recalls, precisions)
    except Exception as e:
        warnings.warn("Failed to calculate AUPRC because {}".format(e))
        auc = 'NA'
    return auc


def compute_mcc(golds_for_eval, probs_for_eval):
    try:
        p = sum(golds_for_eval)
        n = sum([not el for el in golds_for_eval])
        fp, tp, thresholds = _binary_clf_curve(golds_for_eval, probs_for_eval)
        tn, fn = n - fp, p - tp
        mcc = (tp * tn - fp * fn) / (np.sqrt(((tp + fp) * (fp + tn) * (tn + fn) * (fn + tp))) + 1e-10)
    except Exception as e:
        warnings.warn("Failed to calculate MCC because {}".format(e))
        mcc = 'NA'
    return max(mcc)
