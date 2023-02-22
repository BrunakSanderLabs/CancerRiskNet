import json
import pickle
import sys
import os
from os.path import dirname, realpath
sys.path.insert(0, dirname(dirname(dirname(realpath(__file__)))))
from cancerrisknet.utils.parsing import get_code, md5, load_data_settings
from cancerrisknet.utils.date import parse_date
import cancerrisknet.datasets.factory as dataset_factory
from cancerrisknet.datasets.disease_progression import DiseaseProgressionDataset
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib.gridspec import GridSpec
import math
import argparse

#generates two histograms side by side, one with birth as reference and the other as end of date/ pc diagnose
parser = argparse.ArgumentParser(description='Grid Search Results Collector.')
parser.add_argument('--save_dir', type=str, required=True, help="results dir")
parser.add_argument('--exp_id', type=str, required=True, help="exp id")

args = parser.parse_args()
results_path = os.path.join(args.save_dir, args.exp_id) + ".results"
resumed_args = pickle.load(open(results_path, "rb"))
args.__dict__ = resumed_args
SETTINGS = load_data_settings(args)['SETTINGS']

metadata = json.load(open(args.metadata_path))

mean_cancer_age = 66
std_cancer_age = 11.77
norm_dist_cancer_occurrence = np.linspace(-3 * std_cancer_age + mean_cancer_age, 3 * std_cancer_age + mean_cancer_age, 100)

histograms_code_description = SETTINGS.DISEASE_HISTOGRAM #TODO change this for USA visualization

class Disease_Progression_Histogram(DiseaseProgressionDataset):
    def __init__(self, metadata, args):
        self.args = args
        self.metadata = metadata
        self.patients = []
        self.histograms_code = histograms_code_description
        self.patient_summary = {
            "code":[],
            "future_panc_cancer":[],
            "event_to_eod":[],
            "dob_to_event":[]
        }

        for patient in tqdm.tqdm(metadata):
            patient_metadata = {patient: metadata[patient]}
            obs_time_end = parse_date(patient_metadata[patient]['end_of_data'])
            dob = parse_date(patient_metadata[patient]['birthdate'])
            events = self.process_events(patient_metadata[patient]['events'])
            future_panc_cancer, outcome_date = self.get_outcome_date(events, end_of_date=obs_time_end)
            for event in events:
                c = get_code(self.args, event["codes"])
                if c in self.histograms_code:
                    self.patient_summary['code'].append(self.histograms_code[c])
                    event_to_eod = (event['admit_date'] - outcome_date).days/365
                    dob_to_event = (event['admit_date'] - dob).days/365
                    self.patient_summary["future_panc_cancer"].append(future_panc_cancer)
                    self.patient_summary["event_to_eod"].append(event_to_eod)
                    self.patient_summary["dob_to_event"].append(dob_to_event)


summary_data = Disease_Progression_Histogram(metadata, args)
histograms = pd.DataFrame.from_dict(summary_data.patient_summary)
os.makedirs("figures/histograms/", exist_ok=True)

for r,code in enumerate(set(summary_data.histograms_code.values())):
    fig, axes = plt.subplots(1,1, figsize=(8,8))

    df = histograms[histograms.code==code]
    try:
        dob_to_event = pd.DataFrame({'Non cancer patients': df.groupby('future_panc_cancer').get_group(False).dob_to_event,
                    'Cancer patients':   df.groupby('future_panc_cancer').get_group(True).dob_to_event})
        dob_to_event.plot(kind='hist', bins=80, ax=axes, density=1, stacked=False, alpha=.5)
        axes.set(xlim=(-3,110), title=f"{code}")
        axes.plot(norm_dist_cancer_occurrence, 
                stats.norm.pdf(norm_dist_cancer_occurrence, mean_cancer_age, std_cancer_age), 
                alpha=0.5, 
                color='red', 
                label='Cancer diagnosis')
        axes.set_xlabel("Age at Event")
        axes.legend()
    except KeyError:
        pass

    plt.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f"figures/histograms/disease_histogram_{code}.png", bbox_inches='tight')
    plt.close()