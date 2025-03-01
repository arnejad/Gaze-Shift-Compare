import numpy as np
import os
import sys

from modules.dataloader import dataloader, converDataToGazeNet


# insert methods' submodules to be callable inside our code
inner_project_path = os.path.abspath("modules/methods/ACEDNV")
sys.path.insert(0, inner_project_path)
inner_project_path = os.path.abspath("modules/methods/OEMC")
sys.path.insert(0, inner_project_path)


from modules.methods.ACEDNV.modules.scorer import score

### Load Methods
from modules.methods.IVT import ivt
from modules.methods.IDT import idt
from modules.methods.gazeNet.myRun import gazeNet
from modules.methods.remodnav.myRun import pred as remodnav 
# from modules.methods.I2MC.I2MC_api import run as i2mc
from modules.methods.ACEDNV.modules.eventDetector import ACEDNV
from modules.methods.ACEDNV.modules.reader import readDataset as aceReader
from modules.methods.adhoc.resReader import readResult as adhocPreCompPred
from modules.methods.OEMC.online_sim import OnlineSimulator as OEMC_OnlineSimulator
from modules.methods.OEMC.argsProducer import produceArgs as OEMC_ArgsReplicator
from modules.methods.OEMC.preprocessor import Preprocessor as oemc_preprocessor
from modules.utils import evaluate

from config import INP_DIR


# START UNDER DEV.

# update OEMC to predict participants individually
# Warning: SCORES ARE ABOUT 0%
oemc_args = OEMC_ArgsReplicator()
oemc_pproc = oemc_preprocessor(window_length=1,offset=oemc_args.offset,
                                      stride=oemc_args.strides,frequency=250)
oemc_pproc.process_folder(INP_DIR, 'cached/VU')
oemcSimulator = OEMC_OnlineSimulator(oemc_args)
preds, gt = oemcSimulator.simulate(1)
f1_s, f1_e = score(preds, gt, printBool=True)

# FINISHED UNDER DEV.


### Main body of execution

# Note: Different methods might have different dataloaders or different settings for reading

# loading the dataset
data, labels = dataloader(remove_blinks=True, degConv=False) # Note: Different methods have different dataloaders

methods = [
    (idt, {"threshold": 15}),
    (ivt, {"v_threshold": 1.2})
]
f1s, f1e = evaluate(methods, data, labels)

# IM2C algorithm execution
# i2mc_res = i2mc(data[0])


# gazeNet execution
# Warning: SCORES ARE ABOUT 1%
data, labels = dataloader(remove_blinks=False)
df = converDataToGazeNet(data, labels, dummy=False)
f1s, f1e = evaluate([(gazeNet, {})], df, labels)


# RemodNAV method
df = df.drop(['evt', 'status'], axis=1)
remo_res = remodnav(df)
# print(remo_res)


# Adhoc Alg
# TODO: investigate the mismatch why blinks are slightly different in adhoc results and manual labels
adhoc_res, lbls = adhocPreCompPred()
f1_s, f1_e = score(np.concatenate(adhoc_res), np.concatenate(lbls), printBool=True)


# ACE-DNV
ds_x, ds_y = aceReader()       #ACE-DNV's dataloader
f1s, f1e = evaluate([(ACEDNV, {"modelDir": "modules/methods/ACEDNV/model-zoo/random_forest_wb.pkl"})], ds_x, ds_y)


# OEMC
oemc_args = OEMC_ArgsReplicator()
oemc_pproc = oemc_preprocessor(window_length=1,offset=oemc_args.offset,
                                      stride=oemc_args.strides,frequency=250)
oemc_pproc.process_folder(INP_DIR, 'cached/VU')
oemcSimulator = OEMC_OnlineSimulator(oemc_args)
preds, gt = oemcSimulator.simulate(1)
f1_s, f1_e = score(preds, gt)


print("done")


