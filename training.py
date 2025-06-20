import numpy as np
import os
import sys
import pickle

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
from modules.methods.gazeNet.myTrain import main as gazeNet_train
from modules.methods.gazeNet.myRun import predict_LOO as gazeNet_predict
from modules.methods.ACEDNV.modules.eventDetector import ACEDNV_train, ACEDNV
from modules.methods.ACEDNV.modules.reader import readDataset as aceReader
from modules.methods.ranking.resReader import ranking as rankingPreCompPred
from modules.methods.OEMC.online_sim import OnlineSimulator as OEMC_OnlineSimulator
from modules.methods.OEMC.argsProducer import produceArgs as OEMC_ArgsReplicator
from modules.methods.OEMC.preprocessor import Preprocessor as oemc_preprocessor
from modules.methods.OEMC.myRun import runOEMC
from modules.methods.OEMC.myTrain import train_OEMC
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
for p in range(0, len(ds_y)):
    print("leave " + str(p) + " out")
    # Leave-one-out
    x_test = ds_x[p]
    y_test =  ds_y[p]
    x_train = np.array(ds_x)
    x_train = np.delete(ds_x, p, 0)
    y_train = np.array(ds_y)
    y_train = np.delete(ds_y, p, 0)

    ACEDNV_train(x_train, y_train, downSampling="random")    # train the model with all except one

    preds = ACEDNV(x_test, model_dir)      # Test on the left-out recording

    f1s_mi, f1e_mi, cm_s, cm_e = score(preds, y_test, printBool=False)
    f1s_m.append(f1s_mi)
    f1e_m.append(f1e_mi)
    # ash_scores_m.append(ash_score_mi)
    cm_s_all = cm_s_all+cm_s
    cm_e_all = cm_e_all+cm_e
cm_s_avg = np.array(cm_s_all)/(len(ds_y))
cm_e_avg = np.array(cm_e_all)/(len(ds_y))
outputPerformance("ACEDNV-trained", f1s_m, f1e_m, cm_s_avg, cm_e_avg)


#GazeNet

data, labels = dataloader(LABELER, remove_blinks=False, degConv=False)

f1s_m=[] #all f1 scores obtained in for this threshold on all recording
f1e_m=[]
ash_scores_m = []
cm_s_all = [[0,0],[0,0]]
cm_e_all = [[0,0],[0,0]]
modelDir = "/home/ash/projects/Wild-Saccade-Detection-Comparison/modules/methods/gazeNet/logdir/my_model"
print("training GazeNet")
for p in range(0, len(labels)):
    print("leave " + str(p) + " out")
    # Leave-one-out
    x_test = [data[p]]
    y_test =  [labels[p]]
    x_train = data.copy()
    del x_train[p]
    y_train = labels.copy()
    del y_train [p]
    

    train_df = converDataToGazeNet(x_train, y_train, dummy=False, forTrain=True)
    test_df =  converDataToGazeNet(x_test, y_test, dummy=False, forTrain=True)

    gazeNet_train(train_df, str(p)+".pt", model_dir=modelDir, num_epochs=15, num_workers=2, seed=123)

    preds, gts = gazeNet_predict(os.path.join(modelDir, str(p)+".pt"), test_df)

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)

    f1s_mi, f1e_mi, cm_s, cm_e = score(preds, gts, printBool=False)
    f1s_m.append(f1s_mi)
    f1e_m.append(f1e_mi)
    # ash_scores_m.append(ash_score_mi)
    cm_s_all = cm_s_all+cm_s
    cm_e_all = cm_e_all+cm_e
cm_s_avg = np.array(cm_s_all)/(len(labels))
cm_e_avg = np.array(cm_e_all)/(len(labels))
outputPerformance("GazeNet-trained", f1s_m, f1e_m, cm_s_avg, cm_e_avg)


#OEMC

f1s_m=[] #all f1 scores obtained in for this threshold on all recording
f1e_m=[]
ash_scores_m = []
cm_s_all = [[0,0],[0,0]]
cm_e_all = [[0,0],[0,0]]
print("training OEMC")
recs = listRecNames()
for i, p in enumerate(recs):
    print("leave " + str(p) + " out")
    
    train_recs = recs[:i] + recs[i+1:]

    train_OEMC(train_recs, p)
    preds, gts = runOEMC([p], 'modules/methods/OEMC/models/tcn_model_VU_BATCH-2048_LOO-' + p + '.pt', retrained=True)

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)

    f1s_mi, f1e_mi, cm_s, cm_e = score(preds, gts, printBool=False)
    print("sample: " + str(f1s_mi) + " event: " + str(f1e_mi))
    f1s_m.append(f1s_mi)
    f1e_m.append(f1e_mi)
    # ash_scores_m.append(ash_score_mi)
    cm_s_all = cm_s_all+cm_s
    cm_e_all = cm_e_all+cm_e
cm_s_avg = np.array(cm_s_all)/(len(recs))
cm_e_avg = np.array(cm_e_all)/(len(recs))
outputPerformance("OEMC-trained", f1s_m, f1e_m, cm_s_avg, cm_e_avg)



print("done")
