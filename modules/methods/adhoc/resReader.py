from os import listdir, path
from os.path import isfile, join, isdir
from config import INP_DIR
import numpy as np


def readResult():
    recs = [f for f in listdir(INP_DIR) if isdir(join(INP_DIR, f))]
    
    predsAll = []
    
    #  p5_gaze_with_saccades
    for r in recs:
        directory = join(INP_DIR, r)

        preds = np.array(np.genfromtxt(join(directory, r+"_gaze_with_saccades.txt"), delimiter=' ')[:,3], dtype=int)
        
        
        rmidcs = np.where(preds == -1) # remove blinks

        preds = np.delete(preds, rmidcs)
        
        predsAll.append(preds)

    return predsAll
