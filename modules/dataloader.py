
from os import listdir
from os.path import isfile, join, isdir
import numpy as np
from config import INP_DIR, CLOUD_FORMAT, VIDEO_SIZE, DATASET



def _load_PI():
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
        elif 'gaze_positions.csv' in subFiles:
            gazePath = directory+'/gaze_positions.csv'
        else:
            raise Exception("Could not find gaze.csv or gaze_positions.csv file")

        # checking if timestamp.csv exists
        if not  'world_timestamps.csv' in subFiles: raise Exception("Could not find the world_timestamps.csv file")
        timestampPath = directory+'/world_timestamps.csv'

        #read timestamps fileDATASET == "PI":
        tempRead = np.genfromtxt(timestampPath, delimiter=',')

        if CLOUD_FORMAT:
            timestamps = tempRead[1:, 2]
        else:
            timestamps = tempRead[1:, 0]

        #read gaze files
        tempRead = np.genfromtxt(gazePath, delimiter=',')

        if ~CLOUD_FORMAT:
            gazes = tempRead[1:,[3,4]] # extract gaze locations
            #the corrdinate origin is bottom left
            gazes[:,1] = 1 - gazes[:,1]
            gazes = gazes * [VIDEO_SIZE[1], VIDEO_SIZE[0]] #if gaze locations are normalized to 0 to 1, match it with the pixel pixel locations on video

        else:
            gazes = tempRead[1:, 2:]

        T = tempRead[1:, 0] #extract the time points

        frames = tempRead[1:, 1] #extract the corresponding frame number for each timepoint
        frames -= np.min(frames) #set the starting times for the frames to start from 0

        #since each video frame can have multiple gaze locations, select one of them gaze locations
        frames, indcs = np.unique(frames, return_index=True)
        gazeMatch = gazes[indcs]
        # TMatch = T[indcs]

        
        # np.savetxt( r + '_gazeMatch.csv', gazeMatch, delimiter=',')

        ds_x.append(gazes)
    
    return ds_x 



def _load_GiW():
    print("TODO")
    # to work on

def dataloader():
    if DATASET == "PI":
        return _load_PI()
    elif DATASET == "GiW":
        return _load_GiW()
    
    else: raise Exception("The selected dataset is not correct")