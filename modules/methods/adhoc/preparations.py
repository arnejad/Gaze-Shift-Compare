import numpy as np
import saccades,readTimesAndGaze,readIMU
import av,os.path
import cv2 as cv

scaling_factor_velocity=20
extra_black_space_below=120
image=None
mouseX=0
mouseY=0
mouseMoved=False
buttonPressed=False
shown=0
slider=None
prepared='none'
frameInterval=30
folder=''
imageTimes=[]
videoTimes=[]
gazeData=[]
imuData=None
numberOfFrames=1
video=None
last_prepared_folder='select a file'
also_consider_image=None
smoothXgaze=np.array([])
smoothYgaze=np.array([])
horizon=0

def prepareVideo(consider_image_based_gaze):
    global folder,imageTimes,slider,gazeData,imuData,shown,last_prepared_folder,also_consider_image
    if (folder==last_prepared_folder and also_consider_image==consider_image_based_gaze) or folder=='no folder':
        return
    also_consider_image=consider_image_based_gaze
    file=readTimesAndGaze.getImageFile(folder)
    prepareFrameOfFile(file,0)
    showFrameOfFile()
    imageTimes=readTimesAndGaze.getImageTimes(folder)
    slider.tk.config(to=numberOfFrames-1)
    filename=f'{folder}/my_gaze'
    if not os.path.exists(filename):
        print('creating file with processed gaze data')
        saccades.gazeWithSaccadeDetection(folder,consider_image_based_gaze)
    gazeData=np.loadtxt(filename)
    readTimesAndGaze.findOffsets(folder)
    if len(gazeData[0])<5 and consider_image_based_gaze:
        print('adding processed gaze data')
        saccades.gazeWithSaccadeDetection(folder,consider_image_based_gaze)     
        gazeData=np.loadtxt(filename)
    if consider_image_based_gaze:
        gazeData=gazeData[:,(0,1,2,4)]
    else:
        gazeData=gazeData[:,:4]
    filename=f'{folder}/my_imu_data'
    if os.path.exists(filename):
        imuData=np.loadtxt(filename)
    else:
        print('creating file with processed imu data')
        imu=readIMU.readImuData(folder) # gets the raw data
        with open(filename.replace('imu_data','raw_imu_data'),'w') as f:
            for i in range(len(imu)):
                f.write(f'{imu[i,0]} {imu[i,1]} {imu[i,2]} {imu[i,3]} {imu[i,4]} {imu[i,5]} {imu[i,6]}\n')
        imu=readIMU.orientation(imu) # converted first value to head elevation
        imu=readIMU.addSteps(imu) # estimate time of step (non-zero) and its relative magnitude. return only time, elevation and step
        with open(filename,'w') as f:
            for i in range(len(imu)):
                f.write(f'{imu[i,0]} {imu[i,1]} {imu[i,2]} {imu[i,3]}\n')
        imuData=np.loadtxt(filename)
    shown=-99
    last_prepared_folder=folder
    get_estimate_of_elevation_offset()


def prepareFrameOfFile(file,frame):
    global image,video,shown,prepared,frameInterval,numberOfFrames,videoTimes,video
    if file and file!=prepared:
        video=av.open(file)
        numberOfFrames=video.streams.video[0].frames
        frameInterval=int(1000/float(video.streams.video[0].average_rate))
        if os.path.exists(folder+'/my_image_times'):
            with open(folder+'/my_image_times','r') as f:
                videoTimes=[int(values.rstrip()) for values in f.readlines()]
        else:
            print('determining video times')
            videoTimes=[]
            for image in video.decode(video=0):
                videoTimes.append(image.pts)
            with open(folder+'/my_image_times','w') as f:
                for item in videoTimes:
                    f.write(f'{item}\n')
        prepared=file
    elif frame==shown:
        return(False)
    if frame==shown+1:
        videoFrame=next(video.decode(video=0))
    else:    
        video.seek(videoTimes[frame],stream=video.streams.video[0])
        videoFrame=next(video.decode(video=0))
        while videoTimes[frame]>videoFrame.pts:
            videoFrame=next(video.decode(video=0))
    original_image=videoFrame.to_ndarray(format="bgr24")
    image=np.zeros_like(original_image,shape=(1080+extra_black_space_below,1088,3))
    image[0:1080,0:1088,:]=original_image
    shown=frame
    return(True)


def showFrameOfFile():
    global image
    cv.imshow('frame',image)
    cv.setMouseCallback('frame',dealWithMouse)
#    cv.setWindowProperty('frame',cv.WND_PROP_TOPMOST,1)

def dealWithMouse(event,x,y,flags,param):
    global mouseX,mouseY,mouseMoved,buttonPressed
    if event==cv.EVENT_MOUSEMOVE:
        mouseX=x
        mouseY=y
        mouseMoved=True
    if event == cv.EVENT_LBUTTONDOWN:
        buttonPressed=True

def smoothGazeData():
    global smoothXgaze,smoothYgaze
    if len(smoothXgaze)>0:
        return(smoothXgaze,smoothYgaze)
    smoothXgaze=np.loadtxt(f'{folder}/my_smoothed_horizontal_gaze')
    smoothYgaze=np.loadtxt(f'{folder}/my_smoothed_vertical_gaze')
    extra=np.full((len(smoothXgaze),1),-0.5)
    smoothXgaze=np.hstack((smoothXgaze[:,(0,2)],extra))
    smoothYgaze=np.hstack((smoothYgaze[:,(0,2)],extra))
    smoothXgaze[:,1]/=scaling_factor_velocity
    smoothYgaze[:,1]/=scaling_factor_velocity
    return(smoothGazeData())

def get_estimate_of_elevation_offset():
    global horizon
    filename=f'{folder}/estimate of elevation offset'
    if not os.path.exists(filename):
        with open(filename,'w') as file:
            file.write('300\n')
    horizon=np.loadtxt(filename)
    
def save_estimate_of_elevation_offset():
    global horizon
    filename=f'{folder}/estimate of elevation offset'
    with open(filename,'w') as file:
        file.write(f'{horizon}\n')

    
    