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

MODEL_DIR="/home/ash/projects/Wild-Saccade-Detection-Comparison/modules/methods/gazeNet/logdir/model_final"
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

# FILES = []
# for _root, _dir, _files in os.walk('%s/%s'%(args.root, args.dataset)):
#     FILES.extend(['%s/%s' % (_root, _file)
#                   for _file in fnmatch.filter(_files, "*.npy")])

# df = pd.DataFrame(np.random.randint(0,100,size=(20, 3)), columns=['x','y','evt'])
# x_values = np.random.randint(0, 101, size=200)
# y_values = np.random.randint(0, 101, size=200)
# evt_values = np.random.randint(1, 4, size=200)
# df = pd.DataFrame({
#     'x': x_values,
#     'y': y_values,
#     'evt': evt_values
# })
# X_test_all = [df, df]
# #test separate files
# for X_test in tqdm(X_test_all):
    
#     #load data
 
#     _status = np.isnan(X_test['x']) | \
#               np.isnan(X_test['y']) | \
#               ~np.in1d(X_test['evt'], config['events'])
#     X_test['status'] = ~_status
#     test_dataset = EMDataset(config = config, gaze_data = [X_test])
#     n_samples = len(test_dataset)
#     if n_samples<1:
#         continue
#     test_loader = GazeDataLoader(test_dataset, batch_size=config['batch_size'],
#                                  num_workers=1,
#                                  shuffle=False)
#     #predict
#     _gt, _pr, pr_raw = run_infer(model, n_samples, test_loader, **kwargs)

#     #glue back the predictions
#     _data_pr = copy.deepcopy(test_dataset.data)
#     for _d, _pred in zip(_data_pr, pr_raw):
#         _d['evt'] = 0
#         _d['evt'][1:] = np.argmax(_pred, axis=1)+1
#     _data_pr = pd.concat([pd.DataFrame(_d) for _d in _data_pr]).reset_index(drop=True)
#     _data = pd.DataFrame(X_test)
#     _data = _data.merge(_data_pr, on='t', suffixes=('', '_pred'), how='left')
#     _data['evt'] = _data['evt_pred'].replace({np.nan:0})

#     #save
#     etdata_pr.load(_data[['t', 'x', 'y', 'status', 'evt']].values, **{'source':'np_array'})

#     sdir = fdir.replace(args.dataset, '%s_gazeNet'%args.dataset)
#     mkpath(sdir)
#     spath = '%s/%s'%(sdir, fname)
#     etdata_pr.save(spath)
#     etdata_pr.plot(show=False, save=True, spath='%s'%spath)


def pred(X_test):
    _status = np.isnan(X_test['x']) | \
              np.isnan(X_test['y']) | \
              ~np.in1d(X_test['evt'], config['events'])
    X_test['status'] = ~_status
    test_dataset = EMDataset(config = config, gaze_data = [X_test])
    test_loader = GazeDataLoader(test_dataset, batch_size=config['batch_size'],
                                 num_workers=1,
                                 shuffle=False)
    n_samples = len(test_dataset)
    # _gt, _pr, pr_raw = run_infer(model, n_samples, test_loader, **kwargs)
    pr = run_infer(model, n_samples, test_loader, **kwargs)
    return pr