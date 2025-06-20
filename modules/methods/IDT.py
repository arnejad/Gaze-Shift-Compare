############################################################################################
# The code for this algorithm is extracted from 
# https://github.com/mebirtukan/Evaluation-of-eye-movement-event-detection-algorithms/blob/main/I_VT_Classifier.ipynb
# and redistributed under MIT license.

## quotes from the related article: 
# The dispersion threshold can be set to 0.5 to 1∘ of visual angle if the distance from the eye to the screen is known.
# The duration threshold is typically set to a value between 100 and 200 ms depending on task processing demands.

# in the original paper, they refer to the duration threshold, however, it seems in their code that they have not enforced
#this I-DT criterion. The enforce_min_duration function has been added by me to correct for this.
############################################################################################

from scipy.spatial import distance
import csv
import numpy as np
from config import ET_SAMPLING_RATE

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
  # print(Dispersion)
  # print('Max Dipersion=', max(Dispersion))
  # print('min disp=', min(Dispersion))

  for D in Dispersion:
    if(D<disp_thres):
      mvmts.append(0)
    else:
        mvmts.append(1)
  return mvmts
  #store 1 in mvmts[] if dispersion is less than threshold else store 2
  

def enforce_min_duration(labels, min_duration):
    labels = labels.copy()
    i = 0
    while i < len(labels):
        if labels[i] == 0:
            start = i
            while i < len(labels) and labels[i] == 0:
                i += 1
            duration = i - start
            if duration < min_duration:
                labels[start:i] = [1] * duration  # convert to saccade
        else:
            i += 1
    return labels

def idt(data,d_threshold, min_duration_ms):

  samples_per_ms = ET_SAMPLING_RATE / 1000.0
  min_duration_samples = int(round(min_duration_ms * samples_per_ms))
  

  y1=calcu_disp(data, disp_thres=d_threshold)
  
  y_pred=np.array(y1)

  y_pred = enforce_min_duration(y_pred, min_duration_samples)

    #y_pred=np.array(y1)
  y_pred=(y_pred[:-1]) # Remove last sample for alignment

  return y_pred

  
