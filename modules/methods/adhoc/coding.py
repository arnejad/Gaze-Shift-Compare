from guizero import Window,PushButton,Box
import theInterface
import os.path
import shutil

buttons=[]
selected={}
coding_app=None

def draw_for_coding(main_app,foldername):
    global buttons,coding_app
    if coding_app is not None:
        coding_app.show()
        coding_app.enable()
        return
    options=get_options(foldername)
    if len(options[0])==0:
        return
    maxLen=10
    maxNumber=0
    maxLocal=[]
    for column in options:
        if len(column)>maxNumber:
            maxNumber=len(column)
        maxLocal.append(0)
        for line in column:
            if len(line)>maxLocal[-1]:
                maxLocal[-1]=len(line)
    if sum(maxLocal)+3*len(options)>maxLen:
        maxLen=sum(maxLocal)+3*len(options)
    coding_app=Window(main_app,title='coding')
    coding_app.tk.geometry(f'300x{70*maxNumber}+{1088+theInterface.width_on_right}+0')
    coding_app.when_key_pressed=theInterface.deal_with_key_press
    coding_app.text_size=int(500/maxLen)
    n=0
    for c in range(len(options)):
        box=Box(coding_app,width='fill',height='fill',align='left')
        for i in range(len(options[c])):
            buttons.append(PushButton(box,text=options[c][i],command=register_this,args=[n],align='top',width='fill',height='fill'))
            n+=1
    collect_previously_selected(foldername)
    coding_app.repeat(10000,save_any_codes,args=[foldername])
    return

def get_options(foldername):
    filename=f'{foldername}/coding options'
    if os.path.exists(filename):
        options=read_options(filename)
    else:
        if os.path.exists('coding options'):
            options=read_options('coding options')
            shutil.copy('coding options',filename)
        else:
            with open('coding options','w') as file:
                pass
            options=[[]]
    return(options)

def read_options(filename):
    options=[[]]
    with open(filename) as file:
        for line in file:
            if line.startswith('-'):
                options.append([])
            else:
                options[-1].append(line.strip())
    return(options)

def collect_previously_selected(foldername):
    global buttons,selected
    filename=f'{foldername}/selected codes'
    convert={}
    for i in range(len(buttons)):
        convert[buttons[i].text]=i
    if os.path.exists(filename):
        with open(filename) as file:
            for line in file:
                words=line.strip().split()
                where=int(words[0])
                what=' '.join(words[1:])
                if what in convert:
                    if where in selected:
                        selected[where].append(convert[what])
                    else:
                        selected[where]=[convert[what]]

def remove_coding():
    global coding_app
    if coding_app is not None:
        coding_app.disable()
        coding_app.hide()

def save_any_codes(foldername):
    if len(selected)>0:
        outputfile=f'{foldername}/selected codes'
        with open(outputfile,'w') as file:
            for i in sorted(selected):
                for codes in selected[i]:
                    file.write(f'{i} {buttons[codes].text}\n')

def register_this(i):
    if theInterface.frame in selected:
        if i in selected[theInterface.frame]:
            selected[theInterface.frame].remove(i)
            buttons[i].text_color=(0,0,0)
        else:
            selected[theInterface.frame].append(i)
            buttons[i].text_color=(255,0,255)
    else:
        selected[theInterface.frame]=[i]
        buttons[i].text_color=(255,0,255)

def retrieve(frame):
    set_to_black()
    if frame in selected:
        for i in selected[frame]:
            buttons[i].text_color=(255,0,255)
    
def set_to_black():
    for i in range(len(buttons)):
        buttons[i].text_color=(0,0,0)