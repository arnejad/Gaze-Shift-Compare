import subprocess
import os
import shutil

from os import listdir
from os.path import join, isdir
import numpy as np

from modules.utils import cacheLoadedData
from config import MATLAB_PATH, INP_DIR

def runMovingWindow(data, windowSize, lmda): # last param is lambda (the word is reserved in python)

    print("Executing Moving Window Fixation Detection...")
    cacheLoadedData(data)      
    env = os.environ.copy()

    cache_dir = "/home/ash/projects/Wild-Saccade-Detection-Comparison/degs_cached"
    arg_string = f"/home/ash/projects/Wild-Saccade-Detection-Comparison/degs_cached {windowSize} {lmda}"
    env["MATLAB_ARGS"] = arg_string
    script_directory = "modules/methods/Hooge/"
    script_name = "calculateSaccades"
    subprocess.run([MATLAB_PATH, "-batch", f"addpath('{script_directory}'); {script_name}"], env=env)

    # recs = listdir(join(env["MATLAB_ARGS"], "results"))
    recs = [f for f in listdir(INP_DIR) if isdir(join(INP_DIR, f))]
    if len(data) > 30:  #this means that there is only one recording given to the function
        data = [data]
        recs = ['temp']
    predsAll = []
    for r in recs:
        directory = join(cache_dir, "results", r+".csv")
        preds = np.array(np.genfromtxt(directory, delimiter=','), dtype=int)
        predsAll.append(preds)

    shutil.rmtree('degs_cached/results')

    if len(data) > 30:
        return predsAll[0]    
    else:
        return predsAll
    


