from os import listdir, path
from os.path import isfile, join, isdir
import numpy as np

def readData(inputDir, dataset):
    recs = [f for f in listdir(inputDir) if isdir(join(inputDir, f))]
    ds_g = []
    videosList = []
    ds_t = []
    directories = []
        
    for r in recs:
        directory = join(inputDir, r)
        directories.append(directory)
        # list the files inside the input directory
        subFiles = [f for f in listdir(directory) if isfile(join(directory, f))]

        if dataset == "VU":
            gazeData = np.genfromtxt(directory+'/gaze.txt', delimiter=' ')
            ds_g.append(gazeData)
            videosList.append(directory+'/world.mp4')
            

        elif dataset == "DrewsDynamic":
            gazeData_g = np.load(directory+'/gaze.npy')
            gazeData_t = np.load(directory+'/time_gaze.npy')
            gazeData_b = np.zeros(len(gazeData_t))   #no blinks for this dataset
            gazeData = np.column_stack((gazeData_t, gazeData_g, gazeData_b))
            ds_g.append(gazeData)
            videosList.append(directory+'/scene_camera.mp4')
            
            

        times = np.loadtxt(f'{directory}/world image times.txt')
        ds_t.append(times)


    return ds_g, videosList, ds_t, directories
