# This scripts is used to check the environment and existence of necessary files. Python version and package
# dependencies are also checked. If no code map is detected for the json data, one would be generated automatically.
import json
import argparse
import sys
import yaml
import os
from os.path import dirname, realpath
import pkg_resources
import warnings


# Step 1: Check package and update if needed
print("[Step1-CheckFiles][1/3] Checking python environment and version...")
assert sys.version >= '3.7', "[Step1-CheckFiles][1/3] Python version not compatible. v3.7 is required. Abort."

print("[Step1-CheckFiles][1/3] Checking primary location...")
assert 'scripts/Step1-CheckFiles.py' not in os.getcwd(), \
    "[Step1-CheckFiles][1/3] Always run your scripts by using under the project root dir, i.e. not under scripts. Abort."

print("[Step1-CheckFiles][1/3] Checking package dependencies...")
try:
    pkgs = open('requirements.txt', 'r').readlines()
    pkgs = [pkg.rstrip('\b') for pkg in pkgs]
    pkg_resources.require(pkgs)
    print("[Step1-CheckFiles][1/3] All required packages match the desired version.")
except Exception as e:
    print("[Step1-CheckFiles][1/3] One or more packages does not match the desired version. Do `pip install -r "
          "requirements.txt` and then try again.")

print("[Step1-CheckFiles][1/3] Checking CancerRiskNet core framework...")
sys.path.insert(0, dirname(dirname(realpath(__file__))))
try:
    import cancerrisknet
    from cancerrisknet.utils import parsing
except ModuleNotFoundError:
    print("[Step1-CheckFiles][1/3] CancerRiskNet package cannot be found. Check your environment and then continue.")

# Step 2: Check experiment configuration and setting yaml.
print("[Step1-CheckFiles][2/3] Checking setting and configuration...")
parser = argparse.ArgumentParser(description='Perform a pre-launch test and preprocess data if needed')
parser.add_argument("--experiment_config_path", required=True, type=str, help="Path to the search config.")
args = parser.parse_args()
try:
    grid_search_config = json.load(open(args.experiment_config_path, 'r'))
    print("[Step1-CheckFiles][2/3] Experiment config found at {}.".format(args.experiment_config_path))
except FileNotFoundError:
    print("[Step1-CheckFiles][2/3] Experiment config not found. Aborting.")
    sys.exit(1)

known_factor_lists = []
cancer_code_lists = []
settings = []
setting_paths = grid_search_config['search_space']['data_setting_path'] \
    if 'data_setting_path' in grid_search_config['search_space'] else ['data/settings_RPDR.yaml']

for k, path in enumerate(setting_paths):
    idx = "({} out of {})".format(k+1, len(setting_paths))
    print("[Step1-CheckFiles][2/3] Checking data setting yaml files {}...".format(idx))
    try:
        setting = yaml.safe_load(open(path, 'r'))
        settings.append(setting)
        for key in ['PANC_CANCER_CODE', 'END_OF_TIME_DATE']:
            assert key in setting, "[Step1-CheckFiles][3/3]{} An input for {} is required in {}. Aborting.".format(idx, key, path)
            print("[Step1-CheckFiles][2/3]{} Item {} is found in {}.".format(idx, key, path))
        cancer_code_lists.append(setting['PANC_CANCER_CODE'])

        for key in ['ICD8_MAPPER_NAME', 'ICD9_MAPPER_NAME', 'ICD10_MAPPER_NAME']:
            if key not in setting:
                warnings.warn("[Step1-CheckFiles][3/3]{} Warning. Mapper file {} not found.".format(idx, key))
            map_path = setting[key][0]
            assert os.path.exists(map_path), "[Step1-CheckFiles][3/3]{} Mapper file {} not found. Aborting.".format(idx, map_path)
            print("[Step1-CheckFiles][2/3]{} Mapper file {}->{} found.".format(idx, key, map_path))

        if 'KNOWN_RISK_FACTORS' in setting:
            known_factor_lists.append(setting['KNOWN_RISK_FACTORS'])

    except FileNotFoundError:
        print("[Step1-CheckFiles][2/3] Yaml file {} not found. Aborting.".format(path))
        sys.exit(1)


# Step 3: Check experiment data and vocabulary.
print("[Step1-CheckFiles][3/3] Checking data files...")
metadata_paths = grid_search_config['search_space']['metadata']
for k, metadata_path in enumerate(metadata_paths):
    idx = "({} out of {})".format(k + 1, len(metadata_paths))
    print("[Step1-CheckFiles][3/3] Checking metadata and associated vocabulary {}...".format(idx))
    try:
        metadata = json.load(open(metadata_path, 'r'))
    except FileNotFoundError:
        print("[Step1-CheckFiles][3/3]{} Metadata {} not found. Aborting.".format(idx, metadata_path))
        sys.exit(1)

    vocab_path = os.path.join(
        os.path.dirname(metadata_path), os.path.basename(metadata_path).replace('.json', '-vocab.txt')
    )
    if os.path.exists(vocab_path):
        print("[Step1-CheckFiles][3/3]{} The vocabulary for metadata {} found! Checking...".format(idx, metadata_path))
        codes = open(vocab_path, 'r').readlines()
        codes = [c.rstrip('\n') for c in codes]
    else:
        print("[Step1-CheckFiles][3/3]{} For metadata {}, no vocabulary is detected. Automatically generating...".format(
            idx, metadata_path
        ))
        codes = set()
        for pt in metadata:
            codes.update(set([event['codes'] for event in metadata[pt]['events']]))

        codes = list(codes)
        codes.sort()
        with open(vocab_path, 'w') as f:
            [f.write(c + '\n') for c in codes]

    print("[Step1-CheckFiles][3/3]{} For vocabulary {} checking inclusion...".format(idx, vocab_path))
    assert all([known_factors not in codes for known_factors in known_factor_lists]), \
        "[Step1-CheckFiles][3/3]{} For metadata {}, no known_factors are in the vocabulary. Abort."

    cancer_codes = cancer_code_lists[k]
    not_found_codes = [cancer_code for cancer_code in cancer_codes if cancer_code not in codes]
    assert len(not_found_codes) != len(cancer_codes), \
        "[Step1-CheckFiles][3/3]{} For metadata {}, none of the PANC_CANCER_CODE is found in vocabulary. Abort.".format(
            idx, metadata_path
        )

    if not_found_codes:
        warnings.warn("[Step1-CheckFiles][3/3]{} Warning. For metadata {}, one of more PANC_CANCER_CODE not found in vocabulary. "
                      "The missing codes are: {}".format(idx, metadata_path, not_found_codes))


print("[Step1-CheckFiles][DONE] All checks passed! Ready to start training.")
print("[Step1-CheckFiles][NEXT] Proceed to step 2, e.g., do `python scripts/Step2-ModelTrainScheduler.py "
      "--experiment_config_path {}`".format(args.experiment_config_path))
