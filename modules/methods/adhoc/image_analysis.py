# function to estimate probability of saccade between two images: timeImage1 timeImage2 probability
# estimated by comparing mismatch of image at gaze in comparison with mismatch at same position in image
# combined with differnces in distance from best match in image

import readTimesAndGaze
import numpy as np
import cv2 as cv
import av
from typing import List

#breed=1088
#hoog=1080
check=100    # this is the size of the region used to check saccades (half the square size)

def probability_saccade_from_image(folder):
    print(f'analysing images of folder: {folder}')
    file_with_images=readTimesAndGaze.getImageFile(folder)
    imageTimes=readTimesAndGaze.getImageTimes(folder)
    gazeData=readTimesAndGaze.getGazeData(folder)
    video=av.open(file_with_images)
    number_of_frames=video.streams.video[0].frames
    print(f'{number_of_frames} frames')
    frame_interval=1000/float(video.streams.video[0].average_rate)
    print(f'average interval between frames is {frame_interval:.1f} ms')
    probSaccade=np.zeros((len(imageTimes),3))
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
            probSaccade[n,2]=compare(previous,image,prevGaze,gazeCoordinates)
        probSaccade[n,1]=imageTimes[n]
        prevGaze=gazeCoordinates
        previous=image
        videoTimes.append(videoFrame.pts)
    np.savetxt(f'{folder}/my_prob_from_image',probSaccade,fmt='%.4f')
    with open(f'{folder}/my_image_times','w') as f:
        for item in videoTimes:
            f.write(f'{item}\n')
    print('done analysing images')
            

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
    # estimate probability based on how well templates match
    mismatchOldGaze=result[prevGaze[1]-fromY,prevGaze[0]-fromX]
    mismatchNewGaze=result[newGaze[1]-fromY,newGaze[0]-fromX]
    bestYouCanDo=result[y,x]
    sumMisMatch=mismatchOldGaze+mismatchNewGaze-2*bestYouCanDo
    if sumMisMatch==0:
        return(0)
    p_match=(mismatchNewGaze-bestYouCanDo)/sumMisMatch
    # estimate probability based on distance from best fit
    offsetOldGaze=distance(x,y,prevGaze[0]-fromX,prevGaze[1]-fromY)
    offsetNewGaze=distance(x,y,newGaze[0]-fromX,newGaze[1]-fromY)
    p_offset=offsetNewGaze/(offsetOldGaze+offsetNewGaze)
    return(np.sqrt(p_match*p_offset))


def distance(a,b,x,y):
    x-=a
    y-=b
    return(np.sqrt(x*x+y*y))
    
    