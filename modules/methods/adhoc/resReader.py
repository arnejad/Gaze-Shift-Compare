from os import listdir, path
from os.path import isfile, join, isdir
from config import INP_DIR
import numpy as np


def readResult():
    recs = [f for f in listdir(INP_DIR) if isdir(join(INP_DIR, f))]
    
    predsAll = []
    labelsAll = []
    
    #  p5_gaze_with_saccades
    for r in recs:
        directory = join(INP_DIR, r)

        preds = np.array(np.genfromtxt(join(directory, r+"_gaze_with_saccades.txt"), delimiter=' ')[:,3], dtype=int)
        
        labels = np.array(np.genfromtxt(join(directory, r+"_manual coding"), delimiter=' ')[:,1], dtype=int)

       
        rmidcs_lbls = np.where(labels == -1) # remove blinks detected in ground truth

        preds = np.delete(preds, rmidcs_lbls)
        labels = np.delete(labels, rmidcs_lbls)


        rmidcs_sacs = np.where(preds == -1) # remove blinks detected in saccade detection file
        preds = np.delete(preds, rmidcs_sacs)
        labels = np.delete(labels, rmidcs_sacs)

        predsAll.append(preds)
        labelsAll.append(labels)

    return predsAll, labelsAll
