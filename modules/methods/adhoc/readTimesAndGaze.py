# General stuff
import os,re
import msgpack
import numpy as np
import cv2 as cv
from struct import unpack
from os import path

# Finding and converting files
# for gaze columns are
#    time in s
#    x of gaze in degrees (looking to the right is positive)
#    y of gaze in degrees (looking up is positive)
#    confidence in high frequency measurement (1 is high confidence; 0 is none at all)

# some constants
scene_width_pixels=1088
scene_height_pixels=1080
scene_width_deg=82
scene_height_deg=82
# some global values
originalFolderName=''
currentFileName=''
gazeOffsets=[0.0,0.0]

def getFileWith(folder,options):
    for opt in options:
        if opt in os.listdir(folder):
            return(f'{folder}/{opt}')
    print('no files',options,'in '+folder)
    return()

def combine_files(folder,videos):
    newvideo=cv.VideoWriter(f'{folder}/my_combined_world_files.mp4',cv.VideoWriter_fourcc(*'mp4v'),30,(scene_width_pixels,scene_height_pixels))
    for v in videos:
        curr_v=cv.VideoCapture(f'{folder}/{v}')
        while curr_v.isOpened():
            r,frame =curr_v.read()
            if not r:
                break
            newvideo.write(frame)
    newvideo.release()

def put_in_order(names):
    ordered=[]
    if 'world.mp4' in names:
        ordered.append('world.mp4')
    nr=1
    while len(ordered)<len(names):
        for f in names:
            if f'{nr}.mp4' in f:
                ordered.append(f)
        nr+=1
    return(ordered)

def getImageFile(folder):
    files=[f for f in os.listdir(folder) if 'world' in f and '.mp4' in f]
    if 'my_combined_world_files.mp4' in files:
        return(f'{folder}/my_combined_world_files.mp4')
    if len(files)>1:
        combine_files(folder,put_in_order(files))
        return(getImageFile(folder))
    return(f'{folder}/{files[0]}')

def getImageTimes(folder):
    startTime=findStartTime(folder)
    files=[f for f in os.listdir(folder) if 'world' in f and '.time' in f]
    times=np.array([])
    for file in files:
        if len(times)<1:
            times=np.array(read_time(getFileWith(folder,[file])))
        else:
            new_times=np.array(read_time(getFileWith(folder,[file])))
            if new_times[0]>times[0]:
                times=np.concatenate((times,new_times))
            else:
                times=np.concatenate((new_times,times))
    return(times/1000000000-startTime)

def getOriginalName(where):
    with open(getFileWith(where,['info.invisible.json','info.json']),mode='r') as f:
        for line in f:
            item=re.search(r'"recording_id":"([^\"]+)',line)
            if item:
                return(item.group(1))
    return()

def findStartTime(where):
    with open(getFileWith(where,['info.invisible.json','info.json']),mode='r') as f:
        for line in f:
            item=re.search(r'"start_time":(\d+)',line)
            if item:
                return(float(item.group(1))/1000000000)
    return()
           
def findOffsets(where):
    global gazeOffsets
    with open(getFileWith(where,['info.invisible.json','info.json']),mode='r') as f:
        for line in f:
            item=re.search(r'"gaze_offset":\[([^,]+),([^\]]+)',line)
            if item:
                gazeOffsets=[float(item.group(1)),float(item.group(2))]

def getGazeData(folder):
    if path.exists(f'{folder}/gaze_200hz.raw'):
        return(getGaze(folder,(f'{folder}/gaze_200hz.time',f'{folder}/gaze_200hz.raw')))
    elif path.exists(f'{folder}/gaze.pldata'):
        return(getGazePl(folder))
    else:
        return(getGaze(folder,(f'{folder}/gaze ps1.time',f'{folder}/gaze ps1.raw')))

def getGaze(folderName,where):
    startTime=findStartTime(folderName)
    time=np.array(read_time(where[0]),ndmin=2).transpose()
    time=time/1000000000-startTime
    gaze=np.array(read_values(where[1]))
    gaze=np.reshape(gaze,(-1,2))
    gaze=np.hstack((time,gaze))
    return(gaze)

def getGazePl(folder):
    data_gaze=load_pldata_file(f'{folder}/gaze.pldata')
    data=np.empty([len(data_gaze),4])
    for i in range(len(data_gaze)):
        data[i,0]=data_gaze[i]['timestamp']
        data[i,1]=data_gaze[i]['norm_pos'][0]
        data[i,2]=data_gaze[i]['norm_pos'][1]
        data[i,3]=data_gaze[i]['confidence']
    data[:,1]=data[:,1]*scene_width_pixels
    data[:,2]=scene_height_pixels-data[:,2]*scene_height_pixels
    return(data[data[:,3]>0,:3])
    
def load_pldata_file(file):
    data = []
    with open(file,"rb") as fh:
        topic=None
        for topic, payload in msgpack.Unpacker(fh,raw=False,use_list=False):
            datum = serialized_dict_from_msgpack_bytes(payload)
            data.append(datum)
        if topic!='gaze.pi':
            print('wrong topic: ',topic)
            return None
        return data
    
def serialized_dict_from_msgpack_bytes(data):
    return msgpack.unpackb(data,raw=False,use_list=False,ext_hook=msgpack_unpacking_ext_hook)

def msgpack_unpacking_ext_hook(code,data):
    SERIALIZED_DICT_MSGPACK_EXT_CODE = 13
    if code == SERIALIZED_DICT_MSGPACK_EXT_CODE:
        return serialized_dict_from_msgpack_bytes(data)
    return msgpack.ExtType(code,data)

# Get the times from a .time file
def read_time(filename):
    with open(filename,mode='rb') as file:
        inhoud = file.read()
        file.seek(0, 2)
        size = file.tell()
        st='<'+(size//8)*'q'
        data=unpack(st,inhoud)
    return(data)

# Get the values from a data file
def read_values(filename):
    with open(filename,mode='rb') as file:
        inhoud = file.read()
        file.seek(0, 2)
        size = file.tell()
        st='<'+(size//4)*'f'
        data=unpack(st,inhoud)
    return(data)

def getBlinks(folder):
    file=[f for f in os.listdir(folder) if 'blinks' in f and '.csv' in f]
    blinks=[]
    if len(file):
        time=findStartTime(folder)
        with open(folder+'/'+file[0],mode='r') as f:
            for line in f:
                values=line.split(',')
                if values[0]!='section id':
                    blinks.append((float(values[3])/1000000000-time,float(values[4])/1000000000-time))
    return(blinks)
