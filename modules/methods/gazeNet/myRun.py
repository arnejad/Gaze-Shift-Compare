#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 12:40:20 2017

@author: raimondas
"""

#%% imports
import os, sys, glob
from distutils.dir_util import mkpath
# from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()

###
# import copy, argparse, fnmatch

import torch
from modules.methods.gazeNet.utils_lib.etdata import ETData
from modules.methods.gazeNet.utils_lib import utils
from modules.methods.gazeNet.utils_lib.data_loader import EMDataset, GazeDataLoader
from modules.methods.gazeNet.utils_lib.ETeval import run_infer

from modules.methods.gazeNet.model import gazeNET as gazeNET
import modules.methods.gazeNet.model as model_func

#%% functions
# def get_arguments():
#     parser = argparse.ArgumentParser(description='gazeNet: End-to-end eye-movement event detection with deep neural networks')
#     parser.add_argument('root', type=str,
#                         help='Root for datasets')
#     parser.add_argument('dataset', type=str,
#                         help='Dataset')

#     parser.add_argument('--model_dir', type=str, default='model_final',
#                         help='Directory in which to store the logging ')
#     parser.add_argument('--model_name', type=str, default='gazeNET_0004_00003750',
#                         help='Model')

#     parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in dataloading')
#     parser.add_argument('--save_pred', action='store_true', help='Whether to save predictions')
#     return parser.parse_args()

#%% init variables
dev = False
if dev:
    ROOT = './etdata'
    DATASET = 'lund2013_npy_test'

    sys.argv = [sys.argv[0], ROOT, DATASET]
# args = get_arguments()

MODEL_DIR="/home/ash/projects/Wild-Saccade-Detection-Comparison/modules/methods/gazeNet/logdir/model_final/models"
MODEL_NAME = "gazeNET_0004_00003750"

logdir =  os.path.join('logdir', MODEL_DIR)
fname_config = os.path.join(logdir, 'config.json')
if os.path.exists(fname_config):
    configuration = utils.Config(fname_config)
    config = configuration.params
else:
    print("No config file found in %s" % MODEL_DIR)
    sys.exit()

config['split_seqs']=False
config['augment']=False
config['batch_size']=1

cuda = True if config['cuda'] & torch.cuda.device_count() else False
model_name = '%s.pth.tar'%MODEL_NAME

#%% prepare model
num_classes = len(config['events'])
model = gazeNET(config, num_classes)
model_func.load(model, MODEL_DIR, config, model_name)

if cuda:
    model = torch.nn.DataParallel(model).cuda()
model.eval()

#%%testing

print ("Running testing")
kwargs = {
    'cuda': cuda,
    'use_tqdm': False,
    'eval': False,
}
etdata_gt = ETData()
etdata_pr = ETData()

def gazeNet(X_test):
    
    test_dataset = EMDataset(config = config, gaze_data = X_test)
    test_loader = GazeDataLoader(test_dataset, batch_size=config['batch_size'],
                                 num_workers=1,
                                 shuffle=False)
    n_samples = len(test_dataset)
    # _gt, _pr, pr_raw = run_infer(model, n_samples, test_loader, **kwargs)
    pr = run_infer(model, n_samples, test_loader, **kwargs)

    # pr[0] = onlySaccades(pr[0])
    # pr[1] = onlySaccades(pr[1])

    # the second class is saccade (1) but it is given out as (1+1=2)
    pr[0][np.where(np.isin(pr[0], np.array([1, 3, 4, 5, 6])))] = 0
    pr[0][np.where(pr[0]==2)] = 1  

    pr[1][np.where(np.isin(pr[1], np.array([1, 3, 4, 5, 6])))] = 0
    pr[1][np.where(pr[1]==2)] = 1


    return pr




def predict_LOO(model_path, data, device=None, batch_size=1):
    # === 1. Load model checkpoint ===
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    config['split_seqs'] = False
    config['augment'] = False
    config['batch_size'] = batch_size

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 2. Build model with 2 output classes ===
    model = gazeNET(config, num_classes=2)
    # model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Replace final layer (3 → 2 classes), just like in training
    # old_fc_layer = model.fc[0].module  # SequenceWise(Linear)
    # new_fc_layer = torch.nn.Linear(old_fc_layer.in_features, 2, bias=False)

    # with torch.no_grad():
    #     new_fc_layer.weight[0] = 0.5 * (old_fc_layer.weight[0] + old_fc_layer.weight[1])
    #     new_fc_layer.weight[1] = old_fc_layer.weight[2]

    # model.fc[0].module = new_fc_layer

    model = torch.nn.DataParallel(model).to(device)
    model.eval()

    # === 3. Prepare input data ===
    dataset = EMDataset(config=config, gaze_data=data)
    dataloader = GazeDataLoader(dataset, batch_size=config['batch_size'],
                                shuffle=False, num_workers=0)

    # === 4. Run inference ===
    all_predictions = []
    all_gts = []
    with torch.no_grad():
        for inputs, targets, _, _, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=-1).cpu().numpy()  # [B, T]
            targs = targets.cpu()  
            
            all_predictions.extend(preds)
            all_gts.append(targs)
    return all_predictions, all_gts