import readTimesAndGaze
import numpy as np

#from scipy.signal import medfilt
import matplotlib.pyplot as plt


pi=np.arctan2(0,-1)
g=9.80665


# for imu columns are
#    time in s
#    rotation in deg/s around x-axis (rotating to look up is positive)
#    rotation in deg/s around y-axis (rotating to look right is positive)
#    rotation in deg/s around z-axis (tilting head to the right is positive)
#    acceleration in x direction (rightward is positive; m/s^2)
#    acceleration in y direction (downward is positive, so opposing gravity is negative; m/s^2)
#    acceleration in z direction (forward is positive; m/s^2)

def readImuData(folderName):
    startTime=readTimesAndGaze.findStartTime(folderName)
    time=np.array(readTimesAndGaze.read_time(folderName+'/extimu ps1.time'),ndmin=2).transpose()
    time=time/1000000000-startTime
    imu=np.array(readTimesAndGaze.read_values(folderName+'/extimu ps1.raw'))
    imu=np.reshape(imu,(-1,6))
    imu=np.hstack((time,imu))
    imu[:,5]+=0.07
    return(imu)

# provide estimated head elevation (place in 'data' instead of first rotation)
# determine amount of smoothing of estimated orientation from accelerations, by correlating orientation change with rotation
def orientation(data):
    best=0
    value=-1
    dt=np.mean(np.diff(data[:,0]))
    measured_rotation=dt*data[1:,1]
    where=np.argwhere(np.abs(measured_rotation)<1)
    measured_rotation=measured_rotation[where].ravel()
    for n in range(1,100):
        new_estimate=smooth_estimated_elevation(data,n)
        rot=np.diff(new_estimate)
        rot=rot[where].ravel()
        corr=np.corrcoef(rot,measured_rotation)[0,1]
        if corr>value:
            value=corr
            best=n
    print(f'best correlation is {value:.3f} for smoothing with sd={1000*best*dt:.0f} ms')
    n=best
    normal=np.arange(-3*n,3*n+1)/n
    normal=np.exp(-0.5*normal*normal)        
    normal=normal/np.sum(normal)
    new_estimate=np.arctan2(data[:,6],-data[:,5])*180/pi
    new_estimate=np.convolve(new_estimate,normal,'same')
#    rot=np.diff(new_estimate)
#    rot=rot[where].ravel()
#    fig,ax=plt.subplots(figsize=(8,8))
#    ax.scatter(rot,rot2)
#    plt.show()
    elevation=0
    out=[new_estimate[0]]
    for i in range(1,len(data)):
        out.append(0.99*(out[-1]+dt*data[i,1])+0.01*new_estimate[i])
    data[:,1]=np.array(out)
    return(data)

def smooth_estimated_elevation(data,n):
    normal=np.arange(-3*n,3*n+1)/n
    normal=np.exp(-0.5*normal*normal)        
    elevation=np.arctan2(data[:,6],-data[:,5])*180/pi
    normal=normal/np.sum(normal)
    return(np.convolve(elevation,normal,'same'))


minimal_time_between_steps_frames=50 # 50 is 250 ms at 200 hz
# drift to zero to compensate for biases in acceleration and velocity (i.e. velocity and position)
# step is lowest position / recognize stairs by ratio horizontal/forward acceleration at time of step (when walking up)
def addSteps(data):
    angle=data[:,1]*pi/180
    ampl=np.sqrt(data[:,5]*data[:,5]+data[:,6]*data[:,6])
    dirc=np.arctan2(data[:,6],-data[:,5])
    acc=ampl*np.cos(angle-dirc)
    jerk=np.diff(acc)
    where=np.convolve(np.abs(jerk),np.ones(10),'same')/np.convolve(np.abs(jerk),np.ones(10000),'same')
    options=np.argsort(-where)
    # only consider if amongst the largest 20%
    n=int(0.2*len(options))
    step=np.zeros(len(jerk))
    step[options[:n]]=1
    diff=np.diff(step)
    start=np.nonzero(diff==1)[0]+1
    stop=np.nonzero(diff==-1)[0]
    if stop[0]<start[0]:
        stop=stop[1:]
    step=[[0,0]]
    for i in range(len(stop)):
        if start[i]<step[-1][0]+minimal_time_between_steps_frames:
            continue
        s=stop[stop<stop[i]+minimal_time_between_steps_frames][-1]
        peak=start[i]+np.argmax(jerk[start[i]:s+1])
        dal=start[i]+np.argmin(jerk[start[i]:s+1])
        step.append([peak,dal])
    step=np.array(step)
    ampl=jerk[step[:,0]]-jerk[step[:,1]]
    threshold=0.01*np.median(ampl)
    step=step[ampl>threshold,0]
    data[:,2]=acc-np.mean(acc)
    data[:,3]=0
    data[step,3]=jerk[step]
    data[data[:,3]>25*threshold,3]=25*threshold
    data[data[:,3]<0,3]=0
    return(data[:,:4])
