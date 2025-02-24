import numpy as np
import cv2 as cv
import readTimesAndGaze,preparations,theInterface,manual_gaze
import os.path
    
pixToDegX=readTimesAndGaze.scene_width_deg/readTimesAndGaze.scene_width_pixels
pixToDegY=readTimesAndGaze.scene_height_deg/readTimesAndGaze.scene_height_pixels
midX=readTimesAndGaze.scene_width_pixels/2
midY=readTimesAndGaze.scene_height_pixels/2
gazeX=0
gazeY=0
show_horizon=False

def nice_values(low,high):
    global show_digits
    diff=high-low
    if diff<0:
        return(low,high)
    elif diff>1:
        low=float(f'{low-0.5:.0f}')
        high=float(f'{high+0.5:.0f}')
        show_digits=0
        return(low,high)
    show_digits=1
    return(low,high)

def showOtherData(t):
    global show_horizon
    theInterface.drawing.clear()
    plot=theInterface.drawing.rectangle(40,10,theInterface.drawing.tk.winfo_width()-10,theInterface.drawing.tk.winfo_height()-40,color=(255,255,255))
    plot_width=theInterface.drawing.tk.winfo_width()-50
    plot_height=theInterface.drawing.tk.winfo_height()-50
    toPlot=[]
    min_on_plot=1000
    max_on_plot=-1000
    show_horizon=False
    for i in range(len(theInterface.itemsToShow)):
        if theInterface.valuesForItems[i]:
            if theInterface.itemsToShow[i]=='frames':
                frameTimes=preparations.imageTimes[np.nonzero(np.all([preparations.imageTimes>t-theInterface.showTime,preparations.imageTimes<t+theInterface.showTime],axis=0))]
                frameTimes=40+(frameTimes-t+theInterface.showTime)*plot_width/(2*theInterface.showTime)
                for j in frameTimes:
                    theInterface.drawing.line(j,10,j,10+plot_height,color=theInterface.coloursForItems[i],width=1)
            elif theInterface.itemsToShow[i]=='steps':
                imuData=preparations.imuData[np.nonzero(preparations.imuData[:,-1]),:][0]
                imuData=imuData[np.nonzero(np.all([imuData[:,0]>t-theInterface.showTime,imuData[:,0]<t+theInterface.showTime],axis=0)),:][0]
                if len(imuData)>0:
                    imuData[:,0]=40+(imuData[:,0]-t+theInterface.showTime)*plot_width/(2*theInterface.showTime)
                    col=np.max(preparations.imuData[:,-1])
                    for j in imuData:
                        c=int(255*j[-1]/col)
                        theInterface.drawing.line(j[0],10,j[0],10+plot_height,color=(c,255-c,0),width=2)
            elif theInterface.itemsToShow[i]=='head elevation':
                imuData=preparations.imuData[np.nonzero(np.all([preparations.imuData[:,0]>t-theInterface.showTime,preparations.imuData[:,0]<t+theInterface.showTime],axis=0)),:][0]
                if len(imuData)>1:
                    imuData=imuData[:,[0,1]]
                    toPlot.append([imuData,np.amin(preparations.imuData[:,1]),np.amax(preparations.imuData[:,1]),theInterface.coloursForItems[i]])
                    show_horizon=True
            elif theInterface.itemsToShow[i]==f'velocities/{preparations.scaling_factor_velocity}':
                smootheyehor,smootheyever=preparations.smoothGazeData()
                smootheyehor=smootheyehor[np.nonzero(np.all([smootheyehor[:,0]>t-theInterface.showTime,smootheyehor[:,0]<t+theInterface.showTime],axis=0)),:][0]
                smootheyever=smootheyever[np.nonzero(np.all([smootheyever[:,0]>t-theInterface.showTime,smootheyever[:,0]<t+theInterface.showTime],axis=0)),:][0]
                if len(smootheyehor)>1:
                    smootheyehor[:,1]=smootheyehor[:,1]*pixToDegX
                    smootheyever[:,1]=-smootheyever[:,1]*pixToDegY
                    vel=smootheyehor.copy()
                    vel[:,1]=np.sqrt(smootheyehor[:,1]*smootheyehor[:,1]+smootheyever[:,1]*smootheyever[:,1])
                    toPlot.append([vel,0,0,(255,0,255)])
            elif theInterface.itemsToShow[i]=='horizontal gaze':
                eyehor=preparations.gazeData[np.nonzero(np.all([preparations.gazeData[:,0]>t-theInterface.showTime,preparations.gazeData[:,0]<t+theInterface.showTime],axis=0)),:][0]
                if len(eyehor)>1:
                    if theInterface.what_to_show[0].startswith('manual'):
                        eyehor=eyehor[:,[0,1]]
                    else:
                        eyehor=eyehor[:,[0,1,3]]
                    eyehor[:,1]=(eyehor[:,1]-0.5*readTimesAndGaze.scene_width_pixels)*pixToDegX
                    minmax=(preparations.gazeData[:,1]-0.5*readTimesAndGaze.scene_width_pixels)*pixToDegX
                    toPlot.append([eyehor,np.amin(minmax),np.amax(minmax),theInterface.coloursForItems[i]])
            elif theInterface.itemsToShow[i]=='vertical gaze':
                eyever=preparations.gazeData[np.nonzero(np.all([preparations.gazeData[:,0]>t-theInterface.showTime,preparations.gazeData[:,0]<t+theInterface.showTime],axis=0)),:][0]
                if len(eyever)>1:
                    if theInterface.what_to_show[0].startswith('manual'):
                       eyever=eyever[:,[0,2]]
                    else:
                       eyever=eyever[:,[0,2,3]]
                    # vertical is negative because positive in image is down
                    eyever[:,1]=(0.5*readTimesAndGaze.scene_height_pixels-eyever[:,1])*pixToDegY
                    minmax=(0.5*readTimesAndGaze.scene_height_pixels-preparations.gazeData[:,2])*pixToDegY
                    toPlot.append([eyever,np.amin(minmax),np.amax(minmax),theInterface.coloursForItems[i]])
    # fix axes
    for toShow in toPlot:
        lowest=toShow[1]
        if lowest<min_on_plot:
            min_on_plot,max_on_plot=nice_values(lowest,max_on_plot)
        highest=toShow[2]
        if highest>max_on_plot:
            min_on_plot,max_on_plot=nice_values(min_on_plot,highest)
    # draw coding if manually coded
    if len(theInterface.what_to_show)>0 and manual_gaze.gazeCoding is not None:
        coded=manual_gaze.gazeCoding[np.nonzero(np.all([manual_gaze.gazeCoding[:,0]>t-theInterface.showTime,manual_gaze.gazeCoding[:,0]<t+theInterface.showTime],axis=0)),:][0]
        changes=np.nonzero(coded[1:,1]!=coded[:-1,1])[0]
        changes=np.concatenate(([0],changes,[len(coded)-1]))
        fractions=changes/(len(coded)-1)
        if len(changes)>1:
            for i in range(1,len(changes)):
                if coded[changes[i],1]!=0:
                    kleur=np.array(manual_gaze.colours[coded[changes[i],1]-1])
                    theInterface.drawing.rectangle(40+fractions[i-1]*plot_width,10,40+fractions[i]*plot_width,10+plot_height,color=kleur.clip(200,None))
    # draw values on x-axis (time)
    theInterface.drawing.text(40,plot_height+20,f'{t-theInterface.showTime:.3f}')
    theInterface.drawing.text(20+0.5*plot_width,plot_height+20,f'{t:.3f}')
    theInterface.drawing.text(plot_width-5,plot_height+20,f'{t+theInterface.showTime:.3f}')
    theInterface.drawing.line(40+0.5*plot_width,10,40+0.5*plot_width,10+plot_height,color="black",width=1)
    # draw values on y-axis (if a curve is drawn)
    if len(toPlot)>0:
        if show_digits:
            theInterface.drawing.text(5,5,f'{max_on_plot:.2f}')
            theInterface.drawing.text(5,plot_height,f'{min_on_plot:.2f}')
        else:
            theInterface.drawing.text(5,5,f'{max_on_plot:.0f}')
            theInterface.drawing.text(5,plot_height,f'{min_on_plot:.0f}')
        if min_on_plot<0 and max_on_plot>0:
            zero=plot_height*(0-min_on_plot)/(max_on_plot-min_on_plot)
            theInterface.drawing.line(40,10+plot_height-zero,40+plot_width,10+plot_height-zero,color="black",width=1)
            theInterface.drawing.text(5,plot_height-zero,'0')
    # draw curves    
    for toShow in toPlot:
        show=toShow[0]
        show[:,0]=40+(show[:,0]-t+theInterface.showTime)*plot_width/(2*theInterface.showTime)
        show[:,1]=10+plot_height*(1-(show[:,1]-min_on_plot)/(max_on_plot-min_on_plot))
        if len(show[0,:])>2:
            show[show[:,2]-np.append(show[0,2],show[:-1,2])>0,2]=0
            for i in range(1,len(show)):
                if show[i,2]>-1 and show[i-1,2]>-1:
                    theInterface.drawing.line(show[i-1,0],show[i-1,1],show[i,0],show[i,1],width=2+2*max(show[i-1,2],show[i,2]),color=toShow[3])
                else:
                    theInterface.drawing.line(show[i-1,0],show[i-1,1],show[i,0],show[i,1],width=0.25,color=toShow[3])
        else:
            for i in range(1,len(show)):
                theInterface.drawing.line(show[i-1,0],show[i-1,1],show[i,0],show[i,1],width=2,color=toShow[3])


def update_video():
    global pixToDegX,pixToDegY,midX,midY,gazeX,gazeY
    if preparations.prepared=='none':
        return
    if theInterface.playing and theInterface.frame<preparations.numberOfFrames-1:
        theInterface.frame+=1
    if preparations.prepareFrameOfFile(None,theInterface.frame):
        time=preparations.imageTimes[theInterface.frame]
        theInterface.timeInfo.value=f'time: {str(round(time,3))}'
        gaze=np.all([preparations.gazeData[:,0]>time-0.016,preparations.gazeData[:,0]<time+0.016],axis=0)
        gaze=np.ravel(gaze)
        if show_horizon:
            elevation=preparations.imuData[np.all([preparations.imuData[:,0]>time-0.016,preparations.imuData[:,0]<time+0.016],axis=0),1]
            if len(elevation)>0:
                elevation=np.mean(elevation)/pixToDegY
            else:
                elevation=0
        line=[]
        now=True
        for selection in preparations.gazeData[gaze,:]:
            measurement=np.ravel(selection)
            if now:
                if measurement[0]>time:
                    now=False
                    gazeX=measurement[1]-theInterface.add_to_remove_offset[0]
                    gazeY=measurement[2]-theInterface.add_to_remove_offset[1]
                    cv.circle(preparations.image,(int(gazeX),int(gazeY)),20,(0,0,255), thickness=2)
                    theInterface.gazeInfo.value=f'gaze at time: {str(round((gazeX-midX)*pixToDegX,1))} {str(round((midY-gazeY)*pixToDegY,1))}'
                    if show_horizon and not np.isnan(elevation):
                        cv.circle(preparations.image,(int(readTimesAndGaze.scene_width_pixels/2),int(preparations.horizon+elevation)),20,(255,255,255), thickness=2)                        
            line.append([int(measurement[1]-theInterface.add_to_remove_offset[0]),int(measurement[2]-theInterface.add_to_remove_offset[1])])
        cv.polylines(preparations.image,[np.array([line],np.int32).reshape((-1,1,2))],False,(0,0,255),thickness=2)
        preparations.slider.value=theInterface.frame
        showOtherData(time)
    if preparations.mouseMoved:
        x=(preparations.mouseX-midX)*pixToDegX
        y=(midY-preparations.mouseY)*pixToDegY
        theInterface.cursorInfo.value=f'cursor: {str(round(x,1))} {str(round(y,1))}'
        x=(preparations.mouseX-gazeX)*pixToDegX
        y=(gazeY-preparations.mouseY)*pixToDegY
        theInterface.distanceInfo.value=f'distance: {str(round(np.sqrt(x*x+y*y),1))}'
        preparations.mouseMoved=False
    if preparations.buttonPressed:
        if not os.path.exists(preparations.folder+'/my_distances_between_cursor_and_gaze'):
            with open(preparations.folder+'/my_distances_between_cursor_and_gaze',mode='a') as f:
                f.write('frame time gaze_x gaze_y cursor_x cursor_y distance\n')
        if len(theInterface.gazeInfo.value.split(": "))>1:
            with open(preparations.folder+'/my_distances_between_cursor_and_gaze',mode='a') as f:
                f.write(f'{theInterface.frame} {theInterface.timeInfo.value.split(": ")[1]} {theInterface.gazeInfo.value.split(": ")[1]} {theInterface.cursorInfo.value.split(": ")[1]} {theInterface.distanceInfo.value.split(": ")[1]}\n')
            print(f'saved data for frame {theInterface.frame}')
        preparations.buttonPressed=False
    preparations.showFrameOfFile()

theInterface.draw_interface(update_video)
cv.destroyAllWindows()
exit()
