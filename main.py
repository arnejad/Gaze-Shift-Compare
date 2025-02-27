import numpy as np
import pandas as pd
import os
import sys
import torch

from modules.dataloader import dataloader, converDataToGazeNet

# insert methods' submodules to be callable inside our code
inner_project_path = os.path.abspath("modules/methods/ACEDNV")
sys.path.insert(0, inner_project_path)
inner_project_path = os.path.abspath("modules/methods/OEMC")
sys.path.insert(0, inner_project_path)

### Load Methods
from modules.methods.IVT import ivt
from modules.methods.IDT import idt
from modules.methods.gazeNet.myRun import pred as gazeNet
from modules.methods.remodnav.myRun import pred as remodnav 
from modules.methods.I2MC.I2MC_api import run as i2mc
from modules.methods.ACEDNV.modules.eventDetector import pred_detector as acePredictor
from modules.methods.ACEDNV.modules.reader import readDataset as aceReader
from modules.methods.ACEDNV.modules.scorer import score
from modules.methods.adhoc.resReader import readResult as adhocPreCompPred
from modules.methods.OEMC.online_sim import OnlineSimulator as OEMC_OnlineSimulator
from modules.methods.OEMC.argsProducer import produceArgs as OEMC_ArgsReplicator
from modules.methods.OEMC.preprocessor import Preprocessor as oemc_preprocessor

from config import INP_DIR


# START UNDER DEV.



# FINISHED UNDER DEV.


### Main body of execution

# loading the dataset
data, lables = dataloader(remove_blinks=True) # Note: Different methods have different dataloaders

# TODO threshold to be optimized
# IVT algorithm execution
ivt_res = ivt(data[0], v_threshold=0.6)
# f1_s, f1_e = score(ivt_res, lables[0])

# IDT algorithm execution
idt_res = idt(data[0], threshold=0.6)

# IM2C algorithm execution
i2mc_res = i2mc(data[0])


# gazeNet execution
data, lables = dataloader(remove_blinks=False) # Note: Different methods have different dataloaders
df = converDataToGazeNet(data, lables, dummy=False)

gazeNet_res, gazeNet_gt = gazeNet(df)
f1_s, f1_e = score(gazeNet_res, gazeNet_gt)
# print(gazeNet_res)


# RemodNAV method
df = df.drop(['evt', 'status'], axis=1)
remo_res = remodnav(df)
# print(remo_res)


# Adhoc Alg
# TODO: investigate the mismatch why blinks are slightly different in adhoc results and manual labels
adhoc_res, lbls = adhocPreCompPred()

f1_s, f1_e = score(np.concatenate(adhoc_res), np.concatenate(lbls))

# ACE-DNV

ds_x, ds_y = aceReader()
ds_x = np.array(ds_x, dtype=object); 
if ds_y: ds_y = np.array(ds_y, dtype=object)

ace_res = acePredictor(ds_x, ds_y, "modules/methods/ACEDNV/model-zoo/random_forest_wb.pkl")


# OEMC

oemc_args = OEMC_ArgsReplicator()
oemc_pproc = oemc_preprocessor(window_length=1,offset=oemc_args.offset,
                                      stride=oemc_args.strides,frequency=250)
oemc_pproc.process_folder(INP_DIR, 'cached/VU')
oemcSimulator = OEMC_OnlineSimulator(oemc_args)
preds, gt = oemcSimulator.simulate(1)
f1_s, f1_e = score(preds, gt)


print("done")


