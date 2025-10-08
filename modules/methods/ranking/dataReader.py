from os import listdir, path
from os.path import isfile, join, isdir
import numpy as np
from modules.utils import find_long_saccade_segments

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
           

            labels = np.load(directory+'/gt_labels_toggled.npy')
            rmidcs = find_long_saccade_segments(gazeData_t, labels)
            labels[rmidcs] = -1
            # gazeData = np.delete(gazeData, rmidcs, axis=0)
            gazeData_b = labels   #no blinks for this dataset
            gazeData = np.column_stack((gazeData_t, gazeData_g, gazeData_b))

            ds_g.append(gazeData)
            videosList.append(directory+'/scene_camera.mp4')
            
            

        # times = np.loadtxt(f'{directory}/world image times.txt')
        times = np.load(f'{directory}/time_scene_camera.npy')
        ds_t.append(times)


    return ds_g, videosList, ds_t, directories
