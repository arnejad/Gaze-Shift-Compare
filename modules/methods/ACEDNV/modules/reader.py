from os import listdir, path
from os.path import isfile, join, isdir
import numpy as np
import scipy.io
from modules.timeMatcher import timeMatcher
from modules.PatchSimNet import pred_all as patchNet_predAll
from modules.methods.ACEDNV.config import OUT_DIR, VIDEO_SIZE, CLOUD_FORMAT, ACTIVITY_NUM, ACIVITY_NAMES

from modules.preprocess import preprocessor

def zScore_norm(featSet):
    #normalize data (z-score) 
    means = np.mean(featSet, axis=0)
    std = np.std(featSet, axis=0)
    featSet = (featSet-means)/std
    mins = np.min(featSet, axis=0)
    featSet = featSet+ np.abs(mins)
    return featSet


def readDataset(inputDir, dataset, labeler):

    if dataset == "VU":


        # list the folders in the directory
        recs = [f for f in listdir(inputDir) if isdir(join(inputDir, f))]
        ds_x = []
        ds_y = []

        for r in recs:
            # r = "p34"
            # r = '3'

            print("reading data recordng from " + r)
            directory = join(inputDir, r)

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

            #read timestamps file
            timestamps = np.genfromtxt(timestampPath, delimiter=',')

            #read gaze files
            tempRead = np.genfromtxt(gazePath, delimiter=' ')

            if not CLOUD_FORMAT:
                gazes = tempRead[1:,[3,4]]
                #the corrdinate origin is bottom left
                gazes[:,1] = 1 - gazes[:,1]
                gazes = gazes * [VIDEO_SIZE[1], VIDEO_SIZE[0]]
                # tempGazeX = gazes[:,0]
                # tempGazeY = gazes[:,1]
                # gazes = np.column_stack((tempGazeY, tempGazeX))
            else:
                gazes = tempRead[:,[1,2]]

            T = tempRead[:, 0]

            labels = np.array(np.genfromtxt(join(directory, r+"_manual coding_"+labeler), delimiter=' ')[:,1], dtype=int)
            # check if imu data exists.
            gazeMatch, TMatch, frames, lblMatch = timeMatcher(timestamps, gazes, labels)

            # imuMatch = np.array(timeMatcher(timestamps, imu))
            visod = np.loadtxt(join(directory, "visOdom.txt"), delimiter=' ')
            headRot = visod[:,5]
            bodyMotion = visod[:,1:3]

            if not path.exists(join(directory, 'patchDists.csv')): 
                patchDists = patchNet_predAll(vidPath, gazeMatch, frames)
                np.savetxt(join(directory, 'patchDists.csv'), patchDists, delimiter=',')
            else:
                patchDists = np.loadtxt(join(directory, 'patchDists.csv'), delimiter=',')

            patchDists = np.transpose(np.array(patchDists))

            # np.savetxt(OUT_DIR + 'gazeMatch.csv', gazeMatch, delimiter=',')


            feats,lbls = preprocessor(gazes, patchDists, headRot, bodyMotion, T, TMatch, labels, lblMatch)
            
            ds_x.append(feats)
            ds_y.append(lbls)
            # break

    
    elif dataset == "DrewsDynamic":
        
        # list the folders in the directory
        recs = [f for f in listdir(inputDir) if isdir(join(inputDir, f))]
        ds_x = []
        ds_y = []

        for r in recs:
            # r = "p34"
            # r = '3'
            
            # if r in ["F", "I", "J", "K", "L"]:
                # continue

            print("reading data recordng from " + r)
            directory = join(inputDir, r)

            # list the files inside the input directory
            subFiles = [f for f in listdir(directory) if isfile(join(directory, f))]

            # find the video in the input directory
            if 'scene_camera.mp4' in subFiles: 
                vidPath = join(directory, 'scene_camera.mp4')
            else:
                raise Exception("The input directory contains more than one mp4 file")


            # checking if gaze.csv exists
            if 'gaze.npy' in subFiles:
                gazePath = directory+'/gaze.npy'
                gazes = np.load(gazePath)
            else:
                raise Exception("Could not find gaze.csv or gaze_positions.csv file")

            if 'time_optic_flow.npy' in subFiles:
                timestamps = np.load(directory+'/time_visual_similarity.npy')
            else:
                raise Exception("Could not find the world_timestamps.csv file")


            if 'time_gaze.npy' in subFiles:
                T = np.load(directory+'/time_gaze.npy')
            else:
                raise Exception("Could not find the world_timestamps.csv file")
            


            if not CLOUD_FORMAT:
               
                #the corrdinate origin is bottom left
                gazes[:,1] = 1 - gazes[:,1]
                gazes = gazes * [VIDEO_SIZE[1], VIDEO_SIZE[0]]
    


            labels = np.array(np.load(directory+'/gt_labels.npy'), dtype=int)

            # check if imu data exists.
            gazeMatch, TMatch, frames, lblMatch = timeMatcher(timestamps, np.column_stack((T, gazes)), labels)

            # imuMatch = np.array(timeMatcher(timestamps, imu))
            visod = np.loadtxt(join(directory, "visOdom.txt"), delimiter=' ')
            visod = np.delete(visod, (0), axis=0)
            headRot = visod[:,5]
            bodyMotion = visod[:,1:3]

            

            if not path.exists(join(directory, 'patchDists.csv')): 
                patchDists = patchNet_predAll(vidPath, gazeMatch, frames)
                np.savetxt(join(directory, 'patchDists.csv'), patchDists, delimiter=',')
            else:
                patchDists = np.loadtxt(join(directory, 'patchDists.csv'), delimiter=',')

            patchDists = np.transpose(np.array(patchDists))

            # np.savetxt(OUT_DIR + 'gazeMatch.csv', gazeMatch, delimiter=',')
            if len(patchDists) != (len(TMatch)-1):
                print("skipping "+r)
                continue

            feats,lbls = preprocessor(gazes, patchDists, headRot, bodyMotion, T, TMatch, labels, lblMatch)
            
            ds_x.append(feats)
            ds_y.append(lbls)
    
    ds_x = np.array(ds_x, dtype=object); 
    if ds_y: ds_y = np.array(ds_y, dtype=object)
    return ds_x, ds_y