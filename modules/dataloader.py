
from os import listdir
from os.path import isfile, join, isdir
import numpy as np
import pandas as pd
from modules.car2spher import calc as pix2degConv

from config import INP_DIR, CLOUD_FORMAT, VIDEO_SIZE, DATASET



def _load_PI(remove_blinks=False, degConv=False):
    # list the folders in the directory
    recs = [f for f in listdir(INP_DIR) if isdir(join(INP_DIR, f))]
    ds_x = []
    ds_y = []

    for r in recs:
        directory = join(INP_DIR, r)

        # list the files inside the input directory
        subFiles = [f for f in listdir(directory) if isfile(join(directory, f))]

        # find the video in the input directory
        if 'world.mp4' in subFiles: 
            vidPath = join(directory, 'world.mp4')
        else:
            raise Exception("The input directory contains more than one mp4 file")


        # checking if gaze.csv exists
        if 'gaze.csv' in subFiles: 
            gazePath = directory+'/gaze.csv'
        elif "gaze.txt" in subFiles:
            gazePath = directory+'/gaze.txt'
        elif 'gaze_positions.csv' in subFiles:
            gazePath = directory+'/gaze_positions.csv'
        else:
            raise Exception("Could not find gaze.csv or gaze_positions.csv file")

        # checking if timestamp.csv exists
        if not  'world image times.txt' in subFiles: raise Exception("Could not find the world_timestamps.csv file")
        timestampPath = directory+'/world image times.txt'

        timestamps = np.genfromtxt(timestampPath, delimiter=',')


        #read gaze files
        tempRead = np.genfromtxt(gazePath, delimiter=' ')

        gazes = tempRead[:,[1,2]]

        if degConv:
            gazes = pix2degConv(gazes)

        labels = np.array(np.genfromtxt(join(directory, r+"_manual coding"), delimiter=' ')[:,1], dtype=int)

        # T = tempRead[1:, 0] #extract the time points

        # frames = tempRead[1:, 1] #extract the corresponding frame number for each timepoint
        # frames -= np.min(frames) #set the starting times for the frames to start from 0

        # #since each video frame can have multiple gaze locations, select one of them gaze locations
        # frames, indcs = np.unique(frames, return_index=True)
        # gazeMatch = gazes[indcs]
        # # TMatch = T[indcs]

        if remove_blinks:
            rmidcs = np.where(labels == -1) # remove blinks
            labels = np.delete(labels, rmidcs)
            gazes = np.delete(gazes, rmidcs, axis=0)


        # # np.savetxt( r + '_gazeMatch.csv', gazeMatch, delimiter=',')

        ds_x.append(gazes)
        ds_y.append(labels)
        
    return ds_x, ds_y



def _load_GiW():
    print("TODO")
    # to work on

def dataloader(remove_blinks, degConv=False):
    if DATASET == "PI":
        return _load_PI(remove_blinks, degConv)
    elif DATASET == "GiW":
        return _load_GiW()
    
    else: raise Exception("The selected dataset is not correct")


def converDataToGazeNet_old(data, labels, dummy=False):
    
    
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    x_values = np.random.randint(0, 101, size=200)
    y_values = np.random.randint(0, 101, size=200)
    evt_values = np.random.randint(1, 4, size=200)

    if not dummy:
        df = pd.DataFrame({
            'x': data[:, 0],
            'y': data[:, 1],
            'evt': labels
        })
    else:
        df = pd.DataFrame({
        'x': x_values,
        'y': y_values,
        'evt': evt_values
        })

    df = df[df.evt != -1]

    # matching our labels with gazeNet standard
    df.loc[df['evt'] == 1, 'evt'] = 2
    df.loc[df['evt'] == 0, 'evt'] = 1
    

    # df_all = [df]

    return df



def converDataToGazeNet(data, labels, dummy=False):
    
    all_data = []
    
    for i, recording in enumerate(data):
        df = pd.DataFrame({
            'x': recording[:, 0],
            'y': recording[:, 1],
            'evt': labels[i]
        })

        # matching our labels with gazeNet standard
        df.loc[df['evt'] == 1, 'evt'] = 2   # our saccade 1 to gazeNet saccade 2
        df.loc[df['evt'] == 0, 'evt'] = 1   # our fixation 0 to gazeNet fixation 1
        df.loc[df['evt'] == -1, 'evt'] = 3  # our blink -1 to gazeNet blink 3

        _status = np.isnan(df['x']) | \
              np.isnan(df['y']) | \
              ~np.in1d(df['evt'], [1,2,3])
        df['status'] = ~_status    
        

        all_data.append(df)

    # df_all = [df]

    return all_data