import numpy as np
import readTimesAndGaze,image_analysis,os
from os import path

# this function makes and combines others to create the final 'my_gaze' with time x_pixels y_pixels saccade_1_blink_-1
def gazeWithSaccadeDetection(folder,consider_images):
    minimalAmplitudeDeg=1
    minGapBetweenSaccades=0.05 # closer together is not new saccade: so one removed (or combined)
    if consider_images:
        if not os.path.exists(f'{folder}/my_gaze'):
            gazeWithSaccadeDetection(folder,False)
        gazeData=np.loadtxt(f'{folder}/my_gaze')
    else:
        gazeData=readTimesAndGaze.getGazeData(folder)
    vel=evaluateSaccades(folder)
    blinkData=find_blinks(folder)
    # determine values for blinks for each time of gaze
    isBlink=[]
    for t in gazeData[:,0]:
        blink=blinkData[np.logical_and(blinkData[:,0]<=t,blinkData[:,1]>t),0]
        if len(blink)>0:
            isBlink.append(True)
        else:
            isBlink.append(False)
    if consider_images:
        if not path.exists(f'{folder}/my_prob_from_image'):
            image_analysis.probability_saccade_from_image(folder)
        imageData=np.loadtxt(f'{folder}/my_prob_from_image')
        # determine values for images for each time of gaze
        p_image=[]
        for t in gazeData[:,0]:
            value=imageData[np.logical_and(imageData[:,0]<=t,imageData[:,1]>t),2]
            if len(value)>0:
                p_image.append(value[0])
            else:
                p_image.append(0)
        # combine to obtain probability of saccade
        prob=np.sqrt(vel[:,1]*np.array(p_image))
    else:
        prob=vel[:,1]
    prob[prob>0.5]=1
    prob[prob<0.6]=0
    prob[0]=0
    prob[-1]=0
    whereSaccadeStarts=np.nonzero(np.logical_and(prob>0.5,np.roll(prob,1)<0.5))[0]
    whereSaccadeEnds=np.nonzero(np.logical_and(prob>0.5,np.roll(prob,-1)<0.5))[0]
    x=np.loadtxt(f'{folder}/my_smoothed_horizontal_gaze')[:,1].flatten()
    y=np.loadtxt(f'{folder}/my_smoothed_vertical_gaze')[:,1].flatten()
    for i in range(len(whereSaccadeStarts)):
        hor=x[whereSaccadeStarts[i]:whereSaccadeEnds[i]+1]
        ver=y[whereSaccadeStarts[i]:whereSaccadeEnds[i]+1]
        saccHor=hor[-1]-hor[0]
        saccVer=ver[-1]-ver[0]
        dot=(hor[1:]-hor[:-1])*saccHor+(ver[1:]-ver[:-1])*saccVer
        whereWrong=np.nonzero(dot<0)[0]
        whereWrong=[whereSaccadeStarts[i]+w for w in whereWrong]
        prob[whereWrong]=0
    # if too close remove smaller saccade unless they are in the same direction (within 45 deg)
    changed=True
    while changed:
        changed=False
        whereSaccadeStarts=np.nonzero(np.logical_and(prob>0.5,np.roll(prob,1)<0.5))[0]
        whereSaccadeEnds=np.nonzero(np.logical_and(prob>0.5,np.roll(prob,-1)<0.5))[0]
        dx=convertToDegreesX(x[whereSaccadeEnds]-x[whereSaccadeStarts])
        dy=convertToDegreesY(y[whereSaccadeEnds]-y[whereSaccadeStarts])
        length=np.sqrt(dx*dx+dy*dy)
        gap=gazeData[whereSaccadeStarts[1:],0]-gazeData[whereSaccadeEnds[:-1],0]
        for i in range(len(gap)):
            if gap[i]<minGapBetweenSaccades:
                changed=True
                if length[i]>0 and length[i+1]>0:
                    cosAngle=(dx[i]*dx[i+1]+dy[i]*dy[i+1])/(length[i]*length[i+1])
                else:
                    cosAngle=0
                if cosAngle<-0.5*np.sqrt(3):
                    prob[whereSaccadeEnds[i]:whereSaccadeStarts[i+1]]=1
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
    # set to -1 if blink
    prob[isBlink]=-1
    gazeData=np.hstack([gazeData,prob.reshape(-1,1)])
    n=len(gazeData[0])
    if n==4:
        np.savetxt(f'{folder}/my_gaze',gazeData,fmt='%.4f %.2f %.2f %.0f')
    else:
        np.savetxt(f'{folder}/my_gaze',gazeData,fmt='%.4f %.2f %.2f %.0f %.0f')


def distance(x1,y1,x2,y2):
    dx=x2-x1
    dy=y2-y1
    return(np.sqrt(dx*dx+dy*dy))

    
    
def evaluateSaccades(folder):
    # saccade if high rank of velocity or acceleration in smoothed data
    if path.exists(f'{folder}/my_prob_from_gaze'):
        return(np.loadtxt(f'{folder}/my_prob_from_gaze'))
    gazeData=readTimesAndGaze.getGazeData(folder)
    t=gazeData[:,0].flatten()
    if not path.exists(f'{folder}/my_smoothed_horizontal_gaze'):
        np.savetxt(f'{folder}/my_smoothed_horizontal_gaze',polynomial_smoothing(gazeData[:,[0,1]],t,0.02),fmt='%.4f %.2f %.1f %.0f')
    x=np.loadtxt(f'{folder}/my_smoothed_horizontal_gaze')
    if not path.exists(f'{folder}/my_smoothed_vertical_gaze'):
        np.savetxt(f'{folder}/my_smoothed_vertical_gaze',polynomial_smoothing(gazeData[:,[0,2]],t,0.02),fmt='%.4f %.2f %.1f %.0f')
    y=np.loadtxt(f'{folder}/my_smoothed_vertical_gaze')
    h=x[:,2].flatten()
    v=y[:,2].flatten()
    velocity=h*h+v*v
    ranks=velocity.argsort().argsort()
    prob_gaze_vel=np.power(ranks/len(ranks),6.58) # value of exponent chosen so only top 10% of ranks have probability > 0.5
    h=x[:,3].flatten()
    v=y[:,3].flatten()
    acceleration=h*h+v*v
    ranks=acceleration.argsort().argsort()
    prob_gaze_acc=np.power(ranks/len(ranks),6.58) # value of exponent chosen so only top 10% of ranks have probability > 0.5
    np.savetxt(f'{folder}/my_prob_from_gaze',np.column_stack((t,np.maximum(prob_gaze_vel,prob_gaze_acc))),fmt='%.4f')
    return(np.loadtxt(f'{folder}/my_prob_from_gaze'))


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


def find_blinks(folder):
    if path.exists(f'{folder}/my_blinks'):
        return(blinkFileWithRightShape(f'{folder}/my_blinks'))
    blinkData=readTimesAndGaze.getBlinks(folder)
    if len(blinkData)>0:
        print('found a pupil invisible file with blinks')
        np.savetxt(f'{folder}/my_blinks',np.array(blinkData),fmt='%.4f')
        return(blinkFileWithRightShape(f'{folder}/my_blinks'))
    print('estimating blinks from gaze data')
    minimalIntervalBetweenSaccades=0.05
    minimalDurationBlink=0.05
    removeAroundBlink=0.05
    gazeData=readTimesAndGaze.getGazeData(folder)
    interval=(gazeData[-1,0]-gazeData[1,0])/(len(gazeData)-2)
    samples=int(np.round(minimalIntervalBetweenSaccades/interval,0))
    saccadeData=np.loadtxt(f'{folder}/my_prob_from_gaze')
    possible=(saccadeData[:,1]>0.5)
    for i in range(samples):
        possible=np.logical_or(possible,np.logical_and(np.roll(possible,-1),np.roll(possible,samples-i)))
    whereSaccadeStarts=np.nonzero(np.logical_and(possible,np.logical_not(np.roll(possible,1))))[0]
    whereSaccadeEnds=np.nonzero(np.logical_and(possible,np.logical_not(np.roll(possible,-1))))[0]
    while whereSaccadeStarts[0]>whereSaccadeEnds[0]:
        whereSaccadeEnds=whereSaccadeEnds[1:]
    while len(whereSaccadeEnds)<len(whereSaccadeStarts):
        whereSaccadeStarts=whereSaccadeStarts[:-1]
    blinkData=[]
    for i in range(len(whereSaccadeStarts)):
        if gazeData[whereSaccadeEnds[i],0]-gazeData[whereSaccadeStarts[i],0]>minimalDurationBlink:
            dx=gazeData[whereSaccadeEnds[i],1]-gazeData[whereSaccadeStarts[i],1]
            dy=gazeData[whereSaccadeEnds[i],2]-gazeData[whereSaccadeStarts[i],2]
            m=int((whereSaccadeStarts[i]+whereSaccadeEnds[i])/2)
            mx=gazeData[whereSaccadeEnds[i],1]-gazeData[m,1]
            my=gazeData[whereSaccadeEnds[i],2]-gazeData[m,2]
            if mx*mx+my*my>2*(dx*dx+dy*dy):
                blinkData.append([gazeData[whereSaccadeStarts[i],0]-removeAroundBlink,gazeData[whereSaccadeEnds[i],0]+removeAroundBlink])
    np.savetxt(f'{folder}/my_blinks',np.array(blinkData),fmt='%.4f')
    return(blinkFileWithRightShape(f'{folder}/my_blinks'))

def blinkFileWithRightShape(file):
    if os.path.getsize(file)<2:
        return(np.empty((0,2)))
    data=np.loadtxt(file)
    return(data.reshape((-1,2)))

def convertToDegreesX(array):
    return(array.flatten()*readTimesAndGaze.scene_width_deg/readTimesAndGaze.scene_width_pixels)

def convertToDegreesY(array):
    return(array.flatten()*readTimesAndGaze.scene_height_deg/readTimesAndGaze.scene_height_pixels)
