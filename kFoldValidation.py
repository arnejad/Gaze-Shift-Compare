import numpy as np
import os
import sys
import math

from modules.dataloader import dataloader, converDataToGazeNet, listRecNames


# insert methods' submodules to be callable inside our code
inner_project_path = os.path.abspath("modules/methods/ACEDNV")
sys.path.insert(0, inner_project_path)
inner_project_path = os.path.abspath("modules/methods/OEMC")
sys.path.insert(0, inner_project_path)


from modules.methods.ACEDNV.modules.scorer import score

### Load Methods
from modules.methods.gazeNet.myTrain import main as gazeNet_train
from modules.methods.gazeNet.myRun import predict_LOO as gazeNet_predict
from modules.methods.ACEDNV.modules.eventDetector import ACEDNV_train, ACEDNV
from modules.methods.ACEDNV.modules.reader import readDataset as aceReader
from modules.methods.OEMC.myRun import runOEMC
from modules.methods.OEMC.myTrain import train_OEMC
from modules.utils import outputPerformance
from config import INP_DIR, LABELER, DATASET

DATASETS = ["VU-EB", "VU-AG", "D&D"]

METHOD_TO_TRAIN = "OEMC"   #choose between "OEMC", "ACE-DNV", and "GazeNet"

use_ceil=False


#ACE-DNV
if METHOD_TO_TRAIN == "ACE-DNV":
    model_dir = '/home/ash/projects/Wild-Saccade-Detection-Comparison/modules/methods/ACEDNV/model-zoo/gaze-shift.pkl'
    ds_x, ds_y = aceReader(INP_DIR, DATASET, None)       #ACE-DNV's dataloader

    n = len(ds_y)

    test_size = max(1, math.ceil(n/6) if use_ceil else n // 6)

    f1s_m=[] #all f1 scores obtained in for this threshold on all recording
    f1e_m=[]
    ash_scores_m = []
    cm_s_all = [[0,0],[0,0]]
    cm_e_all = [[0,0],[0,0]]
    print("training ACE-DNV")
    for i in range(6):
        start = i * test_size
        end = min((i+1) * test_size, n)
        print("leave " + str(start) + " to "+ str(end)+" out")
        # Leave-one-out
        x_test = ds_x[start:end]
        y_test =  ds_y[start:end]
        x_train = np.concatenate((ds_x[:start], ds_x[end:]), axis=0)
        y_train = np.concatenate((ds_y[:start], ds_y[end:]), axis=0)

        ACEDNV_train(x_train, y_train, model_dir, downSampling="random")    # train the model with all except one

        preds = ACEDNV(x_test, model_dir)      # Test on the left-out recording

        f1s_mi = 0
        f1e_mi = 0
        cm_s_i = [[0,0],[0,0]]
        cm_e_i = [[0,0],[0,0]]
        for t_i, pred in enumerate(preds):
            f1s_mij, f1e_mij, cm_s_ij, cm_e_ij = score(pred, y_test[t_i], printBool=False)
            f1s_mi += f1s_mij
            f1e_mi += f1e_mij
            cm_s_i += cm_s_ij
            cm_e_i += cm_e_ij
        f1s_m.append(f1s_mi/(t_i+1))
        f1e_m.append(f1e_mi/(t_i+1))
        # ash_scores_m.append(ash_score_mi)
        cm_s_all = cm_s_all+(cm_s_i/3)
        cm_e_all = cm_e_all+(cm_s_i/3)
    cm_s_avg = np.array(cm_s_all)/(6)
    cm_e_avg = np.array(cm_e_all)/(6)
    outputPerformance("ACEDNV-trained", f1s_m, f1e_m, cm_s_avg, cm_e_avg)



#GazeNet

if METHOD_TO_TRAIN == "GazeNet":
    data, labels = dataloader(DATASET, INP_DIR, None, remove_blinks=False, degConv=False)



    n = len(labels)

    test_size = max(1, math.ceil(n/6) if use_ceil else n // 6)


    f1s_m=[] #all f1 scores obtained in for this threshold on all recording
    f1e_m=[]
    ash_scores_m = []
    cm_s_all = [[0,0],[0,0]]
    cm_e_all = [[0,0],[0,0]]
    modelDir = "/home/ash/projects/Wild-Saccade-Detection-Comparison/modules/methods/gazeNet/logdir/my_model"
    print("training GazeNet")
    for i in range(6):
        start = i * test_size
        end = min((i+1) * test_size, n)
        print("leave " + str(start) + " to "+ str(end)+" out")
        # Leave-one-out
        x_test = data[start:end]
        y_test =  labels[start:end]
        x_train = data.copy()
        del x_train[start:end]
        y_train = labels.copy()
        del y_train [start:end]
        

        train_df = converDataToGazeNet(x_train, y_train, dummy=False, forTrain=True)
        test_df =  converDataToGazeNet(x_test, y_test, dummy=False, forTrain=True)

        gazeNet_train(train_df, str(start)+".pt", model_dir=modelDir, num_epochs=15, num_workers=2, seed=123)

        preds, gts = gazeNet_predict(os.path.join(modelDir, str(start)+".pt"), test_df)

        preds = np.concatenate(preds)
        gts = np.concatenate(gts)

        f1s_mi, f1e_mi, cm_s, cm_e = score(preds, gts, printBool=False)
        f1s_m.append(f1s_mi)
        f1e_m.append(f1e_mi)
        # ash_scores_m.append(ash_score_mi)
        cm_s_all = cm_s_all+cm_s
        cm_e_all = cm_e_all+cm_e
    cm_s_avg = np.array(cm_s_all)/(6)
    cm_e_avg = np.array(cm_e_all)/(6)
    outputPerformance("GazeNet-trained", f1s_m, f1e_m, cm_s_avg, cm_e_avg)


#OEMC
if METHOD_TO_TRAIN == "OEMC":
    f1s_m=[] #all f1 scores obtained in for this threshold on all recording
    f1e_m=[]
    ash_scores_m = []
    cm_s_all = [[0,0],[0,0]]
    cm_e_all = [[0,0],[0,0]]
    print("training OEMC")
    recs = listRecNames(INP_DIR)
    n = len(recs)
    test_size = max(1, math.ceil(n/6) if use_ceil else n // 6)
    for i in range(6):
        start = i * test_size
        end = min((i+1) * test_size, n)
        
        print("leave " + str(start) + " to "+ str(end)+" out")
        
        # end = start + test_size
        train_recs = recs[:start] + recs[end:]
        test_recs = recs[start:end]
        train_OEMC(train_recs, str(start))
        preds, gts = runOEMC(test_recs, INP_DIR, DATASET, 'modules/methods/OEMC/models/tcn_model_'+DATASET+'_BATCH-2048_LOO-' + str(start) + '.pt', retrained=True)

        preds = np.concatenate(preds)
        gts = np.concatenate(gts)

        f1s_mi, f1e_mi, cm_s, cm_e = score(preds, gts, printBool=False)
        print("sample: " + str(f1s_mi) + " event: " + str(f1e_mi))
        f1s_m.append(f1s_mi)
        f1e_m.append(f1e_mi)
        # ash_scores_m.append(ash_score_mi)
        cm_s_all = cm_s_all+cm_s
        cm_e_all = cm_e_all+cm_e
    cm_s_avg = np.array(cm_s_all)/(6)
    cm_e_avg = np.array(cm_e_all)/(6)
    outputPerformance("OEMC-trained", f1s_m, f1e_m, cm_s_avg, cm_e_avg)



print("done")
