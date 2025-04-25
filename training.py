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
from modules.methods.ACEDNV.modules.eventDetector import ACEDNV_train, ACEDNV
from modules.methods.ACEDNV.modules.reader import readDataset as aceReader
from modules.methods.ranking.resReader import ranking as rankingPreCompPred
from modules.methods.OEMC.online_sim import OnlineSimulator as OEMC_OnlineSimulator
from modules.methods.OEMC.argsProducer import produceArgs as OEMC_ArgsReplicator
from modules.methods.OEMC.preprocessor import Preprocessor as oemc_preprocessor
from modules.methods.OEMC.myRun import runOEMC
from modules.utils import evaluate
from modules.methods.Hooge.run import runMovingWindow
from modules.utils import outputPerformance
from config import INP_DIR, LABELER


#ACE-DNV

model_dir = '/home/ash/projects/Wild-Saccade-Detection-Comparison/modules/methods/ACEDNV/model-zoo/gaze-shift.pkl'
ds_x, ds_y = aceReader(LABELER)       #ACE-DNV's dataloader

f1s_m=[] #all f1 scores obtained in for this threshold on all recording
f1e_m=[]
ash_scores_m = []
cm_s_all = [[0,0],[0,0]]
cm_e_all = [[0,0],[0,0]]
print("training ACE-DNV")
for p in range(1, len(ds_y)):
    print("leave " + str(p) + " out")
    # Leave-one-out
    x_test = ds_x[p]
    y_test =  ds_y[p]
    x_train = np.array(ds_x)
    x_train = np.delete(ds_x, p, 0)
    y_train = np.array(ds_y)
    y_train = np.delete(ds_y, p, 0)

    ACEDNV_train(x_train, y_train)    # train the model with all except one

    preds = ACEDNV(x_test, model_dir)      # Test on the left-out recording

    f1s_mi, f1e_mi, cm_s, cm_e = score(y_test, preds, printBool=False)
    f1s_m.append(f1s_mi)
    f1e_m.append(f1e_mi)
    # ash_scores_m.append(ash_score_mi)
    cm_s_all = cm_s_all+cm_s
    cm_e_all = cm_e_all+cm_e
cm_s_avg = np.array(cm_s_all)/(len(ds_y))
cm_e_avg = np.array(cm_e_all)/(len(ds_y))
outputPerformance("ACEDNV-trained", f1s_m, f1e_m, cm_s_all, cm_e_all)


#OEMC


#GazeNet