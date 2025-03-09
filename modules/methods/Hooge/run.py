import subprocess
import os
from os import listdir
from os.path import join, isdir
import numpy as np

from modules.utils import cacheLoadedData
from config import MATLAB_PATH, INP_DIR

def runHooge(data, labels):

    print("Executing Hooget Fixation Detection...")
    cacheLoadedData(data)
    env = os.environ.copy()
    env["MATLAB_ARGS"] = "/home/ash/projects/Wild-Saccade-Detection-Comparison/degs_cached"
    script_directory = "modules/methods/Hooge/"
    script_name = "calculateSaccades"
    subprocess.run([MATLAB_PATH, "-batch", f"addpath('{script_directory}'); {script_name}"], env=env)

    # recs = listdir(join(env["MATLAB_ARGS"], "results"))
    recs = [f for f in listdir(INP_DIR) if isdir(join(INP_DIR, f))]

    predsAll = []
    for r in recs:
        directory = join(env["MATLAB_ARGS"], "results", r+".csv")
        preds = np.array(np.genfromtxt(directory, delimiter=','), dtype=int)
        predsAll.append(preds)

    return predsAll
    


