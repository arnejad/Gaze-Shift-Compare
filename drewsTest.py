import numpy as np
import os
import sys

from modules.dataloader import dataloader, converDataToGazeNet, listRecNames


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
# from modules.methods.remodnav.myRun import pred as remodnav 
# from modules.methods.I2MC.I2MC_api import run as i2mc
from modules.methods.ACEDNV.modules.eventDetector import ACEDNV
from modules.methods.ACEDNV.modules.reader import readDataset as aceReader
from modules.methods.ranking.resReader import ranking as rankingPreCompPred
from modules.methods.OEMC.online_sim import OnlineSimulator as OEMC_OnlineSimulator
from modules.methods.OEMC.argsProducer import produceArgs as OEMC_ArgsReplicator
from modules.methods.OEMC.preprocessor import Preprocessor as oemc_preprocessor
from modules.methods.OEMC.myRun import runOEMC
from modules.utils import evaluate
from modules.methods.Hooge.run import runMovingWindow
from config import INP_DIR, LABELER, OEMC_MODEL, DATASET


data, labels = dataloader("DrewsDynamic", INP_DIR, None, remove_blinks=True, degConv=False, incTimes=True)
preds = runMovingWindow(data, 6000, 2.7)
evaluate([(runMovingWindow, {})], preds, labels)


data, labels = dataloader("DrewsDynamic", INP_DIR, None, remove_blinks=True, degConv=False)

methods = [
    (ivt, {"v_threshold": 2.0, "min_fixation_duration_ms":15}),
    (idt, {"d_threshold": 26,  "min_duration_ms":15})
]
evaluate(methods, data, labels)



recs = listRecNames(INP_DIR)
preds, gts = runOEMC(recs, INP_DIR, DATASET, OEMC_MODEL)
evaluate([(runOEMC, {})], preds, gts)



##### gazeNet
# Warning: SCORES ARE ABOUT 1%
data, labels = dataloader(DATASET, INP_DIR, None, remove_blinks=False, degConv=False)
df = converDataToGazeNet(data, labels, dummy=False)
evaluate([(gazeNet, {})], df, labels)



#### ACE-DNV
# 1- run video2frames.py to get the video frames
# 2- either run DF-VO, or move the outputs of DF-VO from our published dataset to the folders
ds_x, ds_y = aceReader(INP_DIR, DATASET, None)       #ACE-DNV's dataloader
evaluate([(ACEDNV, {"modelDir": "modules/methods/ACEDNV/model-zoo/random_forest_wb.pkl"})], ds_x, ds_y)

