from guizero import Window,PushButton
import numpy as np
import theInterface
import os.path

buttons=[]
coding_app=None
gazeCoding=None
currentSelection=-1
filename='none yet'
colours={0:(0,0,255),1:(255,0,0),2:(0,255,0),3:(255,0,255),4:(0,255,255),5:(255,255,0),6:(0,0,0)}

def draw_for_coding(main_app,foldername,options):
    global buttons,coding_app,filename,gazeCoding
    if len(options)>7:
        print('you can currently only have up to 7 gaze options')
        options=options[:7]
    if coding_app is not None:
        coding_app.show()
        coding_app.enable()
        return
    maxLen=13
    for line in options:
        if len(line)>maxLen:
            maxLen=len(line)
    coding_app=Window(main_app,title='coding')
    coding_app.tk.geometry(f'300x{70*len(options)}+{1088+theInterface.width_on_right}+0')
    coding_app.when_key_pressed=theInterface.deal_with_key_press
    coding_app.text_size=int(500/maxLen)
    for i in range(len(options)):
        buttons.append(PushButton(coding_app,text=options[i],command=selecting_this,args=[i],align='top',width='fill',height='fill'))
        buttons[-1].text_color=colours[i]
    filename=f'{foldername}/manual coding_'
    if os.path.exists(filename):
        gazeCoding=np.loadtxt(filename)
    else:
        gazeCoding=np.loadtxt(f'{foldername}/my_gaze')
        gazeCoding[:,3]=0
        gazeCoding=gazeCoding[:,(0,3)]
        np.savetxt(filename,gazeCoding,fmt='%.4f %d')
    return

def remove_coding():
    global coding_app
    if coding_app is not None:
        coding_app.disable()
        coding_app.hide()

def selecting_this(i):
    global currentSelection
    if i==currentSelection:
        currentSelection=-1
    else:
        currentSelection=i
    for j in range(len(buttons)):
        if j==currentSelection:
            buttons[j].text_size=int(1.2*coding_app.text_size)
        else:
            buttons[j].text_size=int(0.8*coding_app.text_size)

def set_selection(time_range):
    global gazeCoding,filename
    gazeCoding[np.logical_and(gazeCoding[:,0]>=time_range[0],gazeCoding[:,0]<=time_range[1]),1]=currentSelection+1
    np.savetxt(filename,gazeCoding,fmt='%.4f %d')
    