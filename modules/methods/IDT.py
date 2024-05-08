# This function has been extracted from
# https://github.com/mebirtukan/Evaluation-of-eye-movement-event-detection-algorithms/blob/main/I_VT_Classifier.ipynb


from scipy.spatial import distance
import csv
import numpy as np

sequence_dim = 2


def calcu_disp(data, disp_thres):
  Xs = data[:,[0]]
  Ys = data[:,[1]]

  
  disper = [] #x values difference
  #Y = [] #y values difference 
  #Dispersion=[]
  mvmts=[]

  for i in range(len(data) - 1):
    if i>=sequence_dim:
      value = max((Xs[i-sequence_dim:i+sequence_dim])) - min((Xs[i-sequence_dim:i+sequence_dim]) )+(max(Ys[i-sequence_dim:i+sequence_dim]) - min(Ys[i-sequence_dim:i:i+sequence_dim]))
      disper.append(value[0])
    else:
      disper.append(0)
    #Y.append(max(Ys[i:i+sequence_dim]) - min(Ys[i:i+sequence_dim]) )
  #Dispersion=(X+Y)
  #Dispersion=np.absolute(Dispersion)
  Dispersion=np.absolute(disper)
  print(Dispersion)
  print('Max Dipersion=', max(Dispersion))
  print('min disp=', min(Dispersion))

  for D in Dispersion:
    if(D<disp_thres):
      mvmts.append(0)
    else:
        mvmts.append(1)
  return mvmts
  #store 1 in mvmts[] if dispersion is less than threshold else store 2
  

def idt(data,threshold):
  y1=calcu_disp(data, disp_thres=threshold)
  y_pred=np.array(y1)
    #y_pred=np.array(y1)
  y_pred=(y_pred[:-1])

  return y_pred

  
