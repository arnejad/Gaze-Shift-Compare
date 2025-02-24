from guizero import Window,PushButton,Box,TextBox
options={}


def show(the_app):
    global options
    read_instructions()
    options['done']=[]
    help_app=Window(the_app,title='help',width=1600,height=1000,bg=(120,200,200))
    Box(help_app,width='fill',height=30,align='bottom')
    box=Box(help_app,width='fill',height='fill',align='bottom')
    Box(box,width=50,height='fill',align='left')
    text=TextBox(box,width='fill',height='fill',align='left',multiline=True,scrollbar=True,text="Choose a topic to get information\n\nOr 'done' to return to the program")
    text.text_color=(0,0,200)
    text.bg=(245,245,245)
    text.text_size=18
    Box(box,width=50,height='fill',align='left')
    Box(help_app,width='fill',height=30,align='bottom')
    box=Box(help_app,width='fill',height=60,align='bottom')
    for what in options.keys():
        Box(box,width=10,height='fill',align='left')
        p=PushButton(box,text=what,command=show_this,args=[help_app,what,text],align='left',width='fill',height='fill')
        p.text_size=18
    Box(box,width=10,height='fill',align='left')
    Box(help_app,width='fill',height=30,align='bottom')

def show_this(app,what,where):
    global options
    if what=='done':
        app.hide()
    else:
        where.value='\n\n'.join(options[what])

def read_instructions():
    global options
    key=''
    with open('instructions.txt') as file:
        for line in file:
            if line.startswith('~'):
                key=line[1:].strip()
                options[key]=[]
            elif len(line.strip())>1:
                options[key].append(line.strip())
            