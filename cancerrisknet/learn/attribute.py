import cancerrisknet.learn.train as train
from copy import deepcopy
from cancerrisknet.utils.learn import init_metrics_dictionary, get_dataset_loader, get_train_variables
from collections import defaultdict
from cancerrisknet.utils.parsing import get_code
from tqdm import tqdm
from captum.attr import LayerIntegratedGradients, LayerGradientShap,  TokenReferenceBase, visualization
import torch
import pandas as pd
import numpy as np
from functools import partial


torch.backends.cudnn.enabled = False
# TODO: help needed for documentation

def compute_attribution(attribute_data, model, args):

    model = model.to(args.device)
    test_data_loader = get_dataset_loader(args, attribute_data)
    lig_code = LayerIntegratedGradients(model, model.model.code_embed)

    if hasattr(model.model, 'a_embed_add_fc') and hasattr(model.model, 'a_embed_scale_fc'):
        age_embeddings_layers = [model.model.a_embed_add_fc, model.model.a_embed_scale_fc]
        lig_age = LayerIntegratedGradients(model, age_embeddings_layers)
    else:
        lig_age = None

    test_iterator = iter(test_data_loader)
    word2attr = defaultdict(list)
    word2censor_attr = defaultdict(partial(defaultdict, list))
    try:
        for i, batch in enumerate(tqdm(test_iterator)):
            batch = train.prepare_batch(batch, args)
            codes, attr, ages, add_attr_ages, scale_attr_ages, combined_add_ages = \
                attribute_batch(lig_code, lig_age, batch)

            for patient_codes, patient_attr, gold, days in zip(codes, attr, batch['y'], batch['days_to_censor']):
                patient_codes = patient_codes.split()
                time_bin = int(days//30)
                for c, a in zip(patient_codes, patient_attr[-len(patient_codes):]):
                    code = get_code(args, c)
                    word2attr[code].append(a)
                    if gold:        
                        word2censor_attr[time_bin][code].append(a)
            for patient_age, patient_age_attr in zip(ages, add_attr_ages):
                word2attr["Add-Age-{}".format(patient_age)].append(patient_age_attr)
            for patient_age, patient_age_attr in zip(ages, scale_attr_ages):
                word2attr["Scale-Age-{}".format(patient_age)].append(patient_age_attr)
            for patient_age, patient_age_attr in zip(ages, combined_add_ages):
                word2attr["Combined-Age-{}".format(patient_age)].append(patient_age_attr)
            if i >= args.max_batches_per_dev_epoch:
                break
    except Exception as e:
        print(e)

    return word2attr, word2censor_attr


def attribute_batch(explain_code, explain_age, batch, month_idx=3):
    batch_age = deepcopy(batch)
    if explain_code:
        attributions_code = explain_code.attribute(inputs=(batch['x'], batch['age_seq'], batch['time_seq']),
                                                   n_steps=2,
                                                   return_convergence_delta=False,
                                                   target=month_idx,
                                                   additional_forward_args=batch)
    
        attributions_code = attributions_code.sum(dim=2).squeeze(0)
        attributions_code = attributions_code / torch.norm(attributions_code)
        attributions_code = attributions_code.cpu().detach().numpy()
    else:
        attributions_code = []

    if explain_age:
        attributions_age = explain_age.attribute(inputs=(batch_age['x'], batch_age['age_seq'], batch_age['time_seq']),
                                                 n_steps=2,
                                                 return_convergence_delta=False,
                                                 target=month_idx,
                                                 attribute_to_layer_input=True,
                                                 additional_forward_args=batch_age)

        attributions_age[0] = attributions_age[0].sum(dim=(-1, -2)).squeeze()
        attributions_age[0] = attributions_age[0]/torch.norm(attributions_age[0])
        attributions_age[1] = attributions_age[1].sum(dim=(-1, -2)).squeeze()
        attributions_age[1] = attributions_age[1]/torch.norm(attributions_age[1])

        age_attribution_add = attributions_age[0].cpu().detach().numpy()
        age_attribution_scale = attributions_age[1].cpu().detach().numpy()
        age_attribution_combined = (attributions_age[0] + attributions_age[1]).cpu().detach().numpy()
    else:
        age_attribution_add = []
        age_attribution_scale = []
        age_attribution_combined = []
    
    return batch['code_str'], attributions_code, (batch_age['age']//365).squeeze().tolist(), age_attribution_add,\
        age_attribution_scale, age_attribution_combined
