import subprocess
import os
import shutil

from os import listdir
from os.path import join, isdir
import numpy as np

from modules.utils import cacheLoadedData
from config import MATLAB_PATH, INP_DIR

def runMovingWindow(data, windowSize, lmda): # last param is lambda (the word is reserved in python)
    # moving window impelementation is provided by the authors in MATLAB.
    # to get the results from their code, a process runs their matlab code, it saved the results, and python reads the 
    #    prediction saved on the disk by the matlab script

    print("Executing Moving Window Fixation Detection...")
    cacheLoadedData(data)      
    env = os.environ.copy()
    
    #TODO: change to absoulte path from the os

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    cache_dir = os.path.join(script_dir, "degs_cached")
    arg_string = f"{cache_dir} {windowSize} {lmda}"
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
    elif len(data) ==1:
        return predsAll[0]
    else:
        return predsAll
    


