from guizero import App,Box,Text,PushButton,CheckBox,Slider,Drawing,Combo
import os,preparations,coding,manual_gaze,os.path,shutil,show_help,readTimesAndGaze

width_on_right=600
frame=0
playing=False
showTime=0.5
itemsToShow=['horizontal gaze','vertical gaze',f'velocities/{preparations.scaling_factor_velocity}','frames','steps','head elevation']
valuesForItems=[1,1,0,0,0,0]
coloursForItems=[(255,0,0),(0,0,255),(0,0,0),(220,220,220),(0,0,0),(0,255,255)]
chosenFolder=None
ignoreOffsets=None
add_to_remove_offset=[0,0]
timeInfo=None
gazeInfo=None
cursorInfo=None
distanceInfo=None
drawing=None
checkbox=[]
manual_selection=0
what_to_show=[]
link_for_window=None
box4=None

def draw_interface(function_to_repeat_every_frame_interval):
    global chosenFolder,ignoreOffsets,timeInfo,gazeInfo,cursorInfo,distanceInfo,drawing,checkbox,itemsToShow,valuesForItems,coloursForItems,box4
    preparations.folder,initial_selection=read_latest_folder_used()
    set_saccade_selection(initial_selection)
    my_app=App(title=preparations.folder)
    my_app.tk.geometry(f'{width_on_right}x{1080+preparations.extra_black_space_below}+1088+0')
    my_app.when_key_pressed=deal_with_key_press
    Box(my_app,width='fill',height=10,align='top') # just spacing
    box1=Box(my_app,width='fill',height=30,align='top')
    Box(box1,width=10,height='fill',align='left')
    chosenFolder=PushButton(box1,text=f"use '{preparations.folder.split('/')[-1]}'",command=prepareVideo,args=[my_app],width='fill',align='left')
    Box(box1,width=10,height='fill',align='left')
    PushButton(box1,text='select a different folder',command=get_new_input_folder,args=[my_app],align='left')
    Box(box1,width=10,height='fill',align='left')
    box2=Box(my_app,width='fill',height=30,align='top')
    Box(box2,width=10,height='fill',align='left')
    Text(box2,text='time range',align='left')
    Combo(box2,options=['0.1 sec','0.2 sec','0.5 sec','1 sec','2 sec','5 sec','10 sec'],selected='1 sec',command=set_time_axis,align='left')
    Box(box2,width=20,height='fill',align='left')
    Text(box2,text='show',align='left')
    Combo(box2,options=['simple saccade detection','elaborate saccade detection','manual gaze classification'],selected=initial_selection,command=set_saccade_selection,align='left')
    Box(box2,width=10,height='fill',align='right')
    ignoreOffsets=CheckBox(box2,text='ignore calibration',command=ignore_offsets,align='right')
    box3=Box(my_app,width='fill',height=30,align='top')
    Box(box3,width=10,height='fill',align='left')
    box3a=Box(box3,width=110,height='fill',align='left')
    timeInfo=Text(box3a,text='time:',align='left')
    box3b=Box(box3,width=180,height='fill',align='left')
    gazeInfo=Text(box3b,text='gaze at time:',align='left')
    box3c=Box(box3,width=150,height='fill',align='left')
    cursorInfo=Text(box3c,text='cursor:',align='left')
    box3d=Box(box3,width=100,height='fill',align='left')
    distanceInfo=Text(box3d,text='distance:',align='left')
    play_box=Box(my_app,width='fill',height=50,align='top') # for play/stop and step forward/backward buttons
    Box(play_box,width=10,height='fill',align='left')
    PushButton(play_box,text='play/stop',command=toggle_playing,width='fill',height='fill',align='left')
    Box(play_box,width=10,height='fill',align='left')
    PushButton(play_box,text='previous frame',command=shift_frame,args=[-1],width=10,height='fill',align='left')
    Box(play_box,width=10,height='fill',align='left')
    PushButton(play_box,text='next frame',command=shift_frame,args=[1],width=10,height='fill',align='left')
    Box(play_box,width=10,height='fill',align='left')
    PushButton(play_box,text='?',command=show_help.show,args=[my_app],width=1,height='fill',align='left')
    Box(play_box,width=10,height='fill',align='left')
    box4=Box(play_box,width=50,height='fill',align='left')
    up=PushButton(box4,text='up',width='fill',height='fill',align='top',padx=0,pady=0)
    up.when_left_button_pressed=start_shifting_horizon_up
    up.when_left_button_released=stop_shifting_horizon
    down=PushButton(box4,text='down',width='fill',height='fill',align='top',padx=0,pady=0)
    down.when_left_button_pressed=start_shifting_horizon_down
    down.when_left_button_released=stop_shifting_horizon
    Box(play_box,width=10,height='fill',align='left')
    slider_box=Box(my_app,width='fill',height=50,align='top') # for slider
    preparations.slider=Slider(slider_box,width='fill',command=sliderChanged,start=0,end=100)
    drawing=Drawing(my_app,width='fill',height='fill',align='top')
    drawing.bg=(230,230,230)
    drawing.when_left_button_pressed=press
    drawing.when_left_button_released=release
    selection_box=Box(my_app,width='fill',height=20,align='top') # for selection tickboxes
    Text(selection_box,text=f'{itemsToShow[0]} ',color=coloursForItems[0],align='left')
    Text(selection_box,text=f'{itemsToShow[1]} ',color=coloursForItems[1],align='left')
    for i in range(2,len(itemsToShow)):
        checkbox.append(CheckBox(selection_box,text=itemsToShow[i],command=updateShownItems,args=[i],align='left'))
        checkbox[-1].value=valuesForItems[i]
        checkbox[-1].text_color=coloursForItems[i]
    my_app.text_size=14
    my_app.repeat(preparations.frameInterval,function_to_repeat_every_frame_interval)
    my_app.display()
    coding.save_any_codes(preparations.folder)
    
def press(event):
    global manual_selection
    manual_selection=0
    if what_to_show[0].startswith('manual') and event.x>40 and event.x<drawing.tk.winfo_width()-10 and event.y>10 and event.y<drawing.tk.winfo_height()-40:
        manual_selection=event.x
        drawing.bg=(255,230,230)
    
def release(event):
    global manual_selection
    if manual_selection and event.x>40 and event.x<drawing.tk.winfo_width()-10 and event.y>10 and event.y<drawing.tk.winfo_height()-40:
        total=(drawing.tk.winfo_width()-50)/2
        time=timeInfo.value.split(': ')
        if len(time)>1:
            times=sorted([float(time[1])+showTime*(manual_selection-40-total)/total,float(time[1])+showTime*(event.x-40-total)/total])
            manual_gaze.set_selection(times)
    drawing.bg=(230,230,230)
    preparations.shown=0
    
def prepareVideo(the_app):
    global what_to_show,link_for_window
    preparations.prepareVideo(what_to_show[0].startswith('elaborate'))
    if what_to_show[0].startswith('manual'):
        what_to_show+=get_gaze_options()
        manual_gaze.draw_for_coding(the_app,preparations.folder,what_to_show[1:])
    else:
        coding.draw_for_coding(the_app,preparations.folder)
    link_for_window=the_app
        
def updateShownItems(which):
    valuesForItems[which]=checkbox[which-2].value
    preparations.shown=0
    
def set_time_axis(time):
    global showTime
    showTime=float(time.split(' ')[0])/2
    
def set_saccade_selection(which):
    global what_to_show,link_for_window
    what_to_show=[which]
    if len(preparations.gazeData)!=0:
        if which.startswith("manual gaze"):
            what_to_show+=get_gaze_options()
            coding.remove_coding()
            manual_gaze.draw_for_coding(link_for_window,preparations.folder,what_to_show[1:])
        else:
            manual_gaze.remove_coding()
            coding.draw_for_coding(link_for_window,preparations.folder)
            if which.startswith("elaborate") and len(preparations.gazeData[0]<5):
                preparations.prepareVideo(True)
    with open('where is data','w') as file:
        file.write(f'{preparations.folder}\n{which}\n')


def get_gaze_options():
    if preparations.folder=='no folder':
        return(['fill this later'])
    options=[]
    file=f'{preparations.folder}/gaze options'
    if not os.path.exists(file):
        if not os.path.exists('gaze options'):
            with open('gaze options','w') as f:
                f.write('saccades\nblinks\n')
        shutil.copy('gaze options',file)
    with open(file) as f:
        for line in f:
            options.append(line.strip())
    return(options)
        
def read_latest_folder_used():
    try:
        with open('where is data') as file:
            previous_file='no folder'
            line=file.readline().strip()
            if os.path.exists(line):
                previous_file=line
            line=file.readline().strip()
            if len(line)>1:
                return(previous_file,line)
            else:
                return(previous_file,'simple saccade detection')
    except:
        return('no folder','simple saccade detection')

# select pupil invisible data folder to analyse
def get_new_input_folder(app):
    global playing
    playing=False
    if preparations.folder=='select a folder':
        where=os.getcwd()
    else:
        where='/'.join(preparations.folder.split('/')[:-1])
    f=app.select_folder(title='select a folder',folder=where)
    if len(f):
        with open('where is data','w') as file:
            file.write(f'{f}\n{what_to_show[0]}\n')
        preparations.folder=f
        app.title=f
        chosenFolder.text=f"use '{preparations.folder.split('/')[-1]}'"
    app.update()

def toggle_playing():
    global playing
    if playing:
        playing=False
    else:
        playing=True
        coding.set_to_black()

def sliderChanged(value):
    global frame
    frame=int(value)
    coding.retrieve(frame)
    
def shift_frame(value):
    global frame
    frame+=value
    if frame<0:
        frame=0
    if frame>preparations.numberOfFrames-1:
        frame=preparations.numberOfFrames-1
    preparations.slider.value=frame
    
def ignore_offsets():
    global ignoreOffsets,add_to_remove_offset
    if ignoreOffsets.value==0:
        add_to_remove_offset=[0,0]
    else:
        add_to_remove_offset=readTimesAndGaze.gazeOffsets

def deal_with_key_press(event):
    global frame
    ascii_value=event.tk_event.keycode
    if ascii_value==8124162 or ascii_value==2063660802:
        frame-=1
        preparations.slider.value=frame
    elif ascii_value==8189699 or ascii_value==2080438019:
        frame+=1
        preparations.slider.value=frame

def shift_horizon(value):
    preparations.horizon+=value
    preparations.save_estimate_of_elevation_offset()
    
def stop_shifting_horizon():
    global box4
    box4.cancel(shift_horizon)
    
def start_shifting_horizon_up():
    global box4
    shift_horizon(-1)
    box4.repeat(200,shift_horizon,[-1])
    
def start_shifting_horizon_down():
    global box4
    shift_horizon(1)
    box4.repeat(200,shift_horizon,[1])
