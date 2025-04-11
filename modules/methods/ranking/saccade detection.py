import numpy as np
import os,av
from os import path
from os import walk
import cv2 as cv
from typing import List

scene_width_pixels=1088
scene_height_pixels=1080
scene_width_deg=82
scene_height_deg=82

considerImagesAsWellAsEyeMovements=False

# some choices (not optimized, just guessed)
minimalAmplitudeDeg=1 # a saccade must have an amplitude of at least 1 deg
minGapBetweenSaccades=0.05 # closer together than 50 ms is not a new saccade, so one should be removed or they should be combined
fractionOfNonBlinkTimePossiblySaccade=0.1 # separately for speed and acceleration, so might be close to double
check=100    # this is the size of the region used to check saccades (half the square size in pixels)

def analyse_saccades(folder):
    gazeData=np.loadtxt(f'{folder}/gaze.txt')
    # evaluate gaze motion to estimate probability of being saccade
    prob=evaluateEyeVelocityAndAcceleration(gazeData,folder)
    gazeData=np.column_stack((gazeData,prob))
    # evaluate gaze in image to estimate probability of being saccade
    imageData=evaluateImage(folder)
    # expand values that are per image pair to each time of gaze
    p_image=[]
    for t in gazeData[:,0]:
        value=imageData[np.logical_and(imageData[:,0]<=t,imageData[:,1]>t),2]
        if len(value)>0:
            p_image.append(value[0])
        else:
            p_image.append(0)
    gazeData=np.column_stack((gazeData,p_image))
    # combine to obtain probability of saccade
    if considerImagesAsWellAsEyeMovements:
        prob=np.sqrt(gazeData[:,-2]*gazeData[:,-1])
    else:
        prob=gazeData[:,-2]
    # split into likely times and other
    prob[prob>0.5]=1
    prob[prob<0.6]=0
    # make first and last zero to ease analysis
    prob[0]=0
    prob[-1]=0
    # find potential beginnings and ends of saccades
    whereSaccadeStarts=np.nonzero(np.logical_and(prob>0.5,np.roll(prob,1)<0.5))[0]
    whereSaccadeEnds=np.nonzero(np.logical_and(prob>0.5,np.roll(prob,-1)<0.5))[0]
    # get smoothed gaze positions
    x=np.loadtxt(f'{folder}/my_smoothed_horizontal_gaze')[:,1].flatten()
    y=np.loadtxt(f'{folder}/my_smoothed_vertical_gaze')[:,1].flatten()
    # for each potential saccade
    for i in range(len(whereSaccadeStarts)):
        # isolate positions and determine dot product between each step and total displacement (velocity in global direction)
        hor=x[whereSaccadeStarts[i]:whereSaccadeEnds[i]+1]
        ver=y[whereSaccadeStarts[i]:whereSaccadeEnds[i]+1]
        saccHor=hor[-1]-hor[0]
        saccVer=ver[-1]-ver[0]
        dot=(hor[1:]-hor[:-1])*saccHor+(ver[1:]-ver[:-1])*saccVer
        # remove from potential saccade if direction is reversed
        whereWrong=np.nonzero(dot<0)[0]
        whereWrong=[whereSaccadeStarts[i]+w for w in whereWrong]
        prob[whereWrong]=0
    # if potential saccades are too close to each other in time, remove smaller saccade unless they are in the same direction (within 45 deg) in which case they are combined
    changed=True
    while changed:
        changed=False
        whereSaccadeStarts=np.nonzero(np.logical_and(prob>0.5,np.roll(prob,1)<0.5))[0]
        whereSaccadeEnds=np.nonzero(np.logical_and(prob>0.5,np.roll(prob,-1)<0.5))[0]
        dx=convertToDegreesX(x[whereSaccadeEnds]-x[whereSaccadeStarts])
        dy=convertToDegreesY(y[whereSaccadeEnds]-y[whereSaccadeStarts])
        # determine saccade amplitudes
        length=np.sqrt(dx*dx+dy*dy)
        # determine times between saccades
        gap=gazeData[whereSaccadeStarts[1:],0]-gazeData[whereSaccadeEnds[:-1],0]
        for i in range(len(gap)):
            if gap[i]<minGapBetweenSaccades:
                changed=True
                if length[i]>0 and length[i+1]>0:
                    cosAngle=(dx[i]*dx[i+1]+dy[i]*dy[i+1])/(length[i]*length[i+1])
                else:
                    cosAngle=0
                if cosAngle>0.5*np.sqrt(2):
                    prob[whereSaccadeEnds[i]:whereSaccadeStarts[i+1]]=1 # within 45 deg so combine 
                elif length[i]<length[i+1]:
                    prob[whereSaccadeStarts[i]:whereSaccadeEnds[i]+1]=0
                else:
                    prob[whereSaccadeStarts[i+1]:whereSaccadeEnds[i+1]+1]=0
    # remove saccades smaller than minimal length
    whereSaccadeStarts=np.nonzero(np.logical_and(prob>0.5,np.roll(prob,1)<0.5))[0]
    whereSaccadeEnds=np.nonzero(np.logical_and(prob>0.5,np.roll(prob,-1)<0.5))[0]
    dx=convertToDegreesX(x[whereSaccadeEnds]-x[whereSaccadeStarts])
    dy=convertToDegreesY(y[whereSaccadeEnds]-y[whereSaccadeStarts])
    remove=dx*dx+dy*dy<minimalAmplitudeDeg*minimalAmplitudeDeg
    for i in range(len(remove)):
        if remove[i]:
            prob[whereSaccadeStarts[i]:whereSaccadeEnds[i]+1]=0
    gazeData=np.column_stack((gazeData,prob))
    np.savetxt(f'{folder.split("/")[-1]}_gaze_with_saccades.txt',np.column_stack((gazeData[:,:3],gazeData[:,3]+gazeData[:,-1])),fmt='%.4f %.2f %.2f %.0f')


def distance(x1,y1,x2,y2):
    dx=x2-x1
    dy=y2-y1
    return(np.sqrt(dx*dx+dy*dy))

    
def evaluateEyeVelocityAndAcceleration(gazeData,folder):
    # saccade if high rank of velocity or acceleration in smoothed data
    t=gazeData[:,0].flatten()
    if not path.exists(f'{folder}/my_smoothed_horizontal_gaze'):
        np.savetxt(f'{folder}/my_smoothed_horizontal_gaze',polynomial_smoothing(gazeData[:,[0,1]],t,0.02),fmt='%.6f %.4f %.3f %.2f')
    x=np.loadtxt(f'{folder}/my_smoothed_horizontal_gaze')
    if not path.exists(f'{folder}/my_smoothed_vertical_gaze'):
        np.savetxt(f'{folder}/my_smoothed_vertical_gaze',polynomial_smoothing(gazeData[:,[0,2]],t,0.02),fmt='%.6f %.4f %.3f %.2f')
    y=np.loadtxt(f'{folder}/my_smoothed_vertical_gaze')
    h=x[:,2].flatten()
    v=y[:,2].flatten()
    velocity=h*h+v*v
    blinks=(gazeData[:,3]<0) # find blinks
    exponent=np.log(0.5)/np.log(1-fractionOfNonBlinkTimePossiblySaccade) # value of exponent chosen so only top 10% of ranks have probability > 0.5
    velocity[blinks]=-1  # set speed to -1 if categorized as blink
    ranks=velocity.argsort().argsort()
    fraction=(ranks-blinks.sum())/(len(ranks)-blinks.sum()) # determine fraction of time without blink time
    fraction[fraction<0]=0
    prob_gaze_vel=np.power(fraction,exponent)
    h=x[:,3].flatten()
    v=y[:,3].flatten()
    acceleration=h*h+v*v
    acceleration[blinks]=-1  # set acceleration to -1 if categorized as blink
    ranks=acceleration.argsort().argsort()
    fraction=(ranks-blinks.sum())/(len(ranks)-blinks.sum()) # determine fraction of time without blink time
    fraction[fraction<0]=0
    prob_gaze_acc=np.power(fraction,exponent)
    return(np.maximum(prob_gaze_vel,prob_gaze_acc))


# second order polynomial fit (Savitzky-Golay) for 'values' at times in 'times' considered within certain range
def polynomial_smoothing(values,times,consider_within_range_s):
    rows,columns=values.shape
    if columns != 2:
        print('array with ',columns,' dimensions (not 2) submitted for polynomial smoothing')
        return()
    if rows==0:
        print('no values')
        return()
    output=[]
    for t in times:
        relevant=values[np.where(np.logical_and(values[:,0]>=t-consider_within_range_s,values[:,0]<=t+consider_within_range_s))[0],:]
        if len(relevant)<3:
            output.append([t,0,0,0])
            continue
        relevant=relevant-[t,0]
        p,v,a=fit_second_order_polynomial(relevant)
        output.append([t,p,v,a])
    return(np.array(output))


# actual second order polynomial fit for local use
def fit_second_order_polynomial(data):
    n,c=data.shape
    if c!=2:
        print('wrong number of columns')
        return()
    temp=np.sum(data,axis=0)
    sx=temp.item(0)
    sv=temp.item(1)
    sq=np.multiply(data[:,0],data[:,0])
    sx2=np.sum(sq)
    sx3=np.sum(np.multiply(data[:,0],sq))
    sx4=np.sum(np.multiply(sq,sq))
    sxv=np.sum(np.multiply(data[:,0],data[:,1]))
    sx2v=np.sum(np.multiply(sq,data[:,1]))
    f1=sx4-sx2*sx2/n
    f4=sx3-sx2*sx/n
    f2=2*f4
    f3=2*(sx2v-sx2*sv/n)
    f5=2*(sx2-sx*sx/n)
    f6=2*(sxv-sx*sv/n)
    a=(f3*f5-f2*f6)/(f1*f5-f2*f4)
    v=(f6-a*f4)/f5
    p=(2*sv-a*sx2-2*v*sx)/(2*n)
    return(p,v,a)

def convertToDegreesX(array):
    return(array.flatten()*scene_width_deg/scene_width_pixels)

def convertToDegreesY(array):
    return(array.flatten()*scene_height_deg/scene_height_pixels)


# function to estimate probability of saccade between two images. Output is timeImage1 timeImage2 probability
# estimated by comparing mismatch of image at gaze in comparison with mismatch at same position in image (so if eye had not moved)
# combined with the differnce in distance between the gaze shift and the best match in the image
def evaluateImage(folder):
    if not 'image evaluation values' in os.listdir(folder):
        print(f'analysing images of folder: {folder}')
        file_with_images=getImageFile(folder)
        imageTimes=getImageTimes(folder)
        gazeData=getGazeData(folder)
        video=av.open(file_with_images)
        number_of_frames=video.streams.video[0].frames
        print(f'{number_of_frames} frames')
        frame_interval=1000/float(video.streams.video[0].average_rate)
        print(f'average interval between frames is {frame_interval:.1f} ms')
        probSaccade=np.zeros((len(imageTimes),3)) # three values because first two give time range for easier combination with higher frequency gaze data
        videoTimes:List[int]=[]
        gazeSample=0
        for videoFrame in video.decode(video.streams.video[0]):
            image=videoFrame.to_ndarray(format="bgr24")
            n=len(videoTimes)
            while gazeSample<len(gazeData)-1 and gazeData[gazeSample,0]<imageTimes[n]:
                gazeSample+=1
            gazeCoordinates=np.rint(gazeData[gazeSample,1:]).astype(int)
            if n>0:
                probSaccade[n,0]=imageTimes[n-1]
                probSaccade[n,2]=compare(previous,image,prevGaze,gazeCoordinates)  # the actual analysis
            probSaccade[n,1]=imageTimes[n]
            prevGaze=gazeCoordinates
            previous=image
            videoTimes.append(videoFrame.pts)
        np.savetxt(f'{folder}/image evaluation values',probSaccade)
    return(np.loadtxt(f'{folder}/image evaluation values'))
            

def compare(oldImage,newImage,prevGaze,newGaze):
    shape=oldImage.shape
    if np.all(prevGaze==newGaze):
        return(0.5)
    # make sure region is within image. Oterwise reduce region size.
    fromX=fromY=toX=toY=check
    minx=min(prevGaze[0],newGaze[0])
    maxx=max(prevGaze[0],newGaze[0])
    miny=min(prevGaze[1],newGaze[1])
    maxy=max(prevGaze[1],newGaze[1])
    if minx<check:
        fromX=minx
    if maxx>shape[1]-check:
        toX=shape[1]-maxx
    if miny<check:
        fromY=miny
    if maxy>shape[0]-check:
        toY=shape[0]-maxy
    # if region too small (gaze too far beyond the edge) don't bother
    if fromX+toX<10 or fromY+toY<10:
        return(0.5)
    # find best match of region in old image within new image
    result=cv.matchTemplate(newImage,oldImage[prevGaze[1]-fromY:prevGaze[1]+toY,prevGaze[0]-fromX:prevGaze[0]+toX,:],eval('cv.TM_SQDIFF_NORMED'))
    # get position of best match
    (x,y)=cv.minMaxLoc(result)[2]
    # estimate probability based on how well templates match: likely to be saccade if images at gaze do not match well (in comparison with best possible)
    mismatchOldGaze=result[prevGaze[1]-fromY,prevGaze[0]-fromX]
    mismatchNewGaze=result[newGaze[1]-fromY,newGaze[0]-fromX]
    bestYouCanDo=result[y,x]
    sumMisMatch=mismatchOldGaze+mismatchNewGaze-2*bestYouCanDo
    if sumMisMatch==0:
        return(0)
    p_match=(mismatchNewGaze-bestYouCanDo)/sumMisMatch
    # estimate probability based on distance from best fit: likely to be saccade if image at gaze is far from previous image
    offsetOldGaze=distance(x,y,prevGaze[0]-fromX,prevGaze[1]-fromY)
    offsetNewGaze=distance(x,y,newGaze[0]-fromX,newGaze[1]-fromY)
    p_offset=offsetNewGaze/(offsetOldGaze+offsetNewGaze)
    return(np.sqrt(p_match*p_offset))


def getImageFile(folder):
    files=[f for f in os.listdir(folder) if 'world' in f and '.mp4' in f]
    return(f'{folder}/{files[0]}')

def getImageTimes(folder):
    return(np.loadtxt(f'{folder}/world image times.txt'))

def getGazeData(folder):
    return(np.loadtxt(f'{folder}/gaze.txt'))


for (dirpath, dirnames, filenames) in walk('selected part'):
    for file in filenames:
        if 'gaze.txt' in file:
            analyse_saccades(dirpath) # all you need to do is to give the correct folder-name here. The rest of this part is just finding it
            print(dirpath.split('/')[-1])

