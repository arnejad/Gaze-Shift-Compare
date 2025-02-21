import os,av
import numpy as np
from os import walk


def check(folder,file):
    video=av.open(f'{folder}/{file}')
    number_of_frames=video.streams.video[0].frames
    data=np.loadtxt(f'{folder}/world image times.txt')
    print(folder,number_of_frames,data.shape[0])
    return
    

check("/media/ash/Expansion/data/Saccade-Detection-Methods/p34", "world.mp4")
            
# for (dirpath,dirnames,filenames) in walk('selected part'):
#     file=[f for f in filenames if 'world' in f and '.mp4' in f]
#     if len(file)>0:
#         check(dirpath,file[0])
# exit(0)         
