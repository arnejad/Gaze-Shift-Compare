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
from config import INP_DIR, LABELER, DATASET

DATASETS = ["VU", "VU", "D&D"]
DATASET_DIRS = ['/media/ash/Expansion/data/Saccade-Detection-Methods', 
                '/media/ash/Expansion/data/Saccade-Detection-Methods',
                "/media/ash/Expansion/data/drews-dynamic"]
LABELERS = ["EB", "AG", None]

METHOD_TO_TRAIN = "OEMC"   #choose between "OEMC", "ACE-DNV", and "GazeNet"

use_ceil=False


#ACE-DNV
if METHOD_TO_TRAIN == "ACE-DNV":
    

    for i, ds_train in enumerate(DATASETS):
        model_dir = '/home/ash/projects/Wild-Saccade-Detection-Comparison/modules/methods/ACEDNV/model-zoo/gaze-shift-' + str(LABELERS[i])+ '.pkl'
        x_train, y_train = aceReader(DATASET_DIRS[i], ds_train, LABELERS[i])       #ACE-DNV's dataloader
        ACEDNV_train(x_train, y_train, model_dir, downSampling="random")    # train the model with all except one

        for j, ds_test in enumerate(DATASETS):
            
            print("Train set: "+ds_train+"-"+str(LABELERS[i])+"  Test set: "+ds_test+"-"+str(LABELERS[j]))

            x_test, y_test = aceReader(DATASET_DIRS[j], ds_test, LABELERS[j])
                      

            preds = ACEDNV(x_test, model_dir)      
            
            preds = np.concatenate(preds)
            gts = np.concatenate(y_test)

            f1s_m, f1e_m, cm_s_all, cm_e_all = score(preds, gts, printBool=False)

            # f1s_mi = 0
            # f1e_mi = 0
            # cm_s_i = [[0,0],[0,0]]
            # cm_e_i = [[0,0],[0,0]]
            # for t_i, pred in enumerate(preds):
            #     f1s_mij, f1e_mij, cm_s_ij, cm_e_ij = score(pred, y_test[t_i], printBool=False)
            #     f1s_mi += f1s_mij
            #     f1e_mi += f1e_mij
            #     cm_s_i += cm_s_ij
            #     cm_e_i += cm_e_ij
            # f1s_m = (f1s_mi/(t_i+1))
            # f1e_m = (f1e_mi/(t_i+1))
           
            # cm_s_all = (cm_s_i/(t_i+1))
            # cm_e_all = (cm_e_i/(t_i+1))
            outputPerformance(METHOD_TO_TRAIN+"-trained-"+str(LABELERS[i])+"-"+str(LABELERS[j]), f1s_m, f1e_m, cm_s_all, cm_e_all)


#GazeNet

if METHOD_TO_TRAIN == "GazeNet":

    modelDir = "/home/ash/projects/Wild-Saccade-Detection-Comparison/modules/methods/gazeNet/logdir/my_model/"
    x_train, y_train = dataloader(ds_train, DATASET_DIRS[i], LABELERS[i], remove_blinks=False, degConv=False)
    x_train = converDataToGazeNet(x_train, y_train, dummy=False, forTrain=True)
    gazeNet_train(x_train, str(LABELERS[i])+".pt", model_dir=modelDir, num_epochs=15, num_workers=2, seed=123)

    for i, ds_train in enumerate(DATASETS):
        for j, ds_test in enumerate(DATASETS):
            print("Train set: "+ds_train+"-"+str(LABELERS[i])+"  Test set: "+ds_test+"-"+str(LABELERS[j]))
            
            x_test, y_test = dataloader(ds_test, DATASET_DIRS[j], LABELERS[j], remove_blinks=False, degConv=False) 
            x_test =  converDataToGazeNet(x_test, y_test, dummy=False, forTrain=True)        

            preds, gts = gazeNet_predict(os.path.join(modelDir, str(LABELERS[i])+".pt"), x_test)

            preds = np.concatenate(preds)
            gts = np.concatenate(gts)

            f1s_m, f1e_m, cm_s_all, cm_e_all = score(preds, gts, printBool=False)

            outputPerformance(METHOD_TO_TRAIN+"-trained-"+str(LABELERS[i])+"-"+str(LABELERS[j]), f1s_m, f1e_m, cm_s_all, cm_e_all)


#OEMC
if METHOD_TO_TRAIN == "OEMC":
    
    
    
    for i, ds_train in enumerate(DATASETS):

        train_recs = listRecNames(DATASET_DIRS[i])
        train_OEMC(train_recs, str(LABELERS[i]), ds_train, LABELERS[i], DATASET_DIRS[i])
        
        if i == 0: continue

        for j, ds_test in enumerate(DATASETS):
            # if ds_test == "VU": continue
            print("Train set: "+ds_train+"-"+str(LABELERS[i])+"  Test set: "+ds_test+"-"+str(LABELERS[j]))

            test_recs = listRecNames(DATASET_DIRS[j])
            if ds_test == "D&D": ds_test = "DrewsDynamic"
            preds, gts = runOEMC(test_recs, DATASET_DIRS[j], ds_test, 'modules/methods/OEMC/models/tcn_model_'+ds_train+'_BATCH-2048_LOO-' + str(LABELERS[i]) + '.pt', retrained=True)

            preds = np.concatenate(preds)
            gts = np.concatenate(gts)

            f1s_m, f1e_m, cm_s_all, cm_e_all = score(preds, gts, printBool=False)

            outputPerformance(METHOD_TO_TRAIN+"-trained-"+str(LABELERS[i])+"-"+str(LABELERS[j]), f1s_m, f1e_m, cm_s_all, cm_e_all)



print("done")
