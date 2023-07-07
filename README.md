# Pancreatic cancer risk predicted from disease trajectories using deep learning

## Introduction
The repository contains the code implementation used for the paper [A deep learning algorithm to predict risk of pancreatic cancer from disease trajectories](https://www.nature.com/articles/s41591-023-02332-5).
We used disease trajectories from EHR to calculate the risk of developing pancreatic cancer at different intervals after the assessment. The repository supports different deep learning [models](cancerrisknet/models).

## Usage
### Data structure

In the data folder you can find a synthetic data with the same structure of the data used in the study. In case you want to reproduce the results on another dataset you need to generate a json file having the same structure. The events do not need to be sorted by 'admdate'.

```json
{
    "PID_0":{
        "birtdate":"1900-01-01",
        "end_of_data":"2022-01-01",
        "events":[
            {
                "admdate":"2000-01-01",
                "admid":"00000000",
                "codes":"E10",
            },
            {
                "admdate":"1990-01-01",
                "admid":"00000001",
                "codes":"C25",
            }
        ]
    },


    
    "PID_9999":{
        "birtdate":"1900-01-01",
        "end_of_data":"2022-01-01",
        "events":[
            {
                "admdate":"1980-01-01",
                "admid":"00000002",
                "codes":"K54",
            },
            {
                "admdate":"2010",
                "admid":"00000004",
                "codes":"C10",
            }
        ]
    }
}
```

## STEP 1
In order to run any experiment, a [config file](configs/sample_grid_search.json) file needs to be saved under the [configs](configs) folder. Check the possible arguments in the [parsing file](cancerrisknet/utils/parsing.py) 

```json
{
  "search_space": {
    "metadata":["data/sample_diag_data.json"],
    "train":[true],
    "dev":[true],
    "test":[false],
    "epochs":[2],
    "num_workers":[8],
    "cuda":[true],
    "hidden_dim":[256],
    "model_name": ["transformer", "gru", "bow"],
    "optimizer": ["adam"],
    "init_lr":[1e-03, 1e-04],
    "train_batch_size": [4],
    "eval_batch_size": [4],
    "max_batches_per_train_epoch": [5],
    "max_batches_per_dev_epoch": [5],
    "max_events_length": [1000],
    "max_eval_indices": [10],
    "pad_size": [200],
    "eval_auroc": [true],
    "eval_auprc": [true],
    "eval_c_index": [true],
    "data_setting_path": ["data/settings_sample_data.yaml"]
  },
  "available_gpus": [1]
}
```
In the [data](data) folder a YAML file must be saved with the following information:
    - PANC_CANCER_CODE [required]: used to define the outcome
    - END_OF_TIME_DATE [required]: date of the last possible record in the data 
    - KNOWN_RISK_FACTORS [optional]: used to run experiment using uniquely the known risk factors 
    - ICD8_MAPPER_NAME, ICD9_MAPPER_NAME, ICD10_MAPPER_NAME [optional]: files uses in the visualization to transform disease codes to plain text. 

Before running any experiments few checks on the requirements, along with constructing the vocabulary for the models, have to be done running the following command:
```
python scripts/Step1-CheckFiles.py --experiment_config_path configs/sample_grid_search.json
``` 
Resolve the possible error and warnings raised by the script before proceeding with the next step. 

## STEP 2

The command used to run experiments is:
```
python scripts/Step2-ModelTrainScheduler --experiment_config_path configs/sample_grid_search.json --search_name sample_search --scheduler single_node_scheduler
```

The --scheduler single_node_scheduler (default) runs the experiment sequentially on the machine where the script is launched. An initial support is also available for torque/moab and google cloud (see --help).

## STEP 3
At the end of Step2 output you can find the command you should probably run to collect the grid search you just run.

```
python -u scripts/Step3-CollectSearchResults.py \
    --experiment_config_path configs/sample_grid_search.json \
    --search_dir searches/untitled-search_d94b4902_20220210-2247 \
    --result_dir results/untitled-search_d94b4902_20220210-2247
```
Note that the folder name inside ```searches``` and ```results```  (in the example example ```untitled-search_d94b4902_20220210-2247```) is generated using the --search_name argument used in Step2 (untitiled-search is default), a short MD5 extracted from the specific parameter for that exact grid search and the datetime. In this way each hyperparameter search will never generate the same folder with the same name. 

If the collection returns some warning (i.e. missing test because the --test flag was not used in the search config) it is possible to rerun Step2 using the in search config the parameter --resume_from_result, which will reload the model and arguments and will run --train / --dev/ --test accordingly to the new config. 

## STEP 4

The final step to collect the results is to generate a table with all the metrics (including CI) running the command below.

```
python scripts/Step4-ResultBootstrap.py --search_metadata searches/untitled-search_d94b4902_20220210-2247/performance_table.csv
``` 

The file performance_table.csv is the output file generated in the Step3 and contains all the experiment that will be used in Step4 to generate the new table with metrics/confidence intervals/curves coordinates. Use the flag --filename to specify the name of the Step4 output. 
