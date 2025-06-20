
############################################################################################
# The code for this algorithm is extracted from 
# https://github.com/mebirtukan/Evaluation-of-eye-movement-event-detection-algorithms/blob/main/I_VT_Classifier.ipynb
# and redistributed under MIT license.


# The saccade recall reaches 98% and the fixation recall 
# is 25% at the lowest velocity threshold value of 0.1 px/ms because, at this threshold value, most points are classified as saccades.
# Therefore, in this case, the optimum velocity threshold value for I-VT is 0.5 px/ms. 
############################################################################################


import numpy as np
from config import ET_SAMPLING_RATE
# In the origin code they assumed that the frequency is 500Hz so there is 2ms gap between every two samples
# we changed 2ms to read from the config file


def enforce_min_duration(labels, min_duration_ms, sampling_rate):
    labels = labels.copy()
    min_samples = int(round((min_duration_ms / 1000.0) * sampling_rate))

    i = 0
    while i < len(labels):
        if labels[i] == 0:  # start of potential fixation
            start = i
            while i < len(labels) and labels[i] == 0:
                i += 1
            duration = i - start
            if duration < min_samples:
                labels[start:i] = [1] * duration  # reclassify as saccade
        else:
            i += 1
    return labels


def ivt(data,v_threshold, min_fixation_duration_ms):
  t_dist = (1/ET_SAMPLING_RATE)*1000
  Xs = data[:,[0]]
  Ys = data[:,[1]]

  diffX = [] #x values difference
  diffY = [] #y values difference 

  for i in range(len(data) - 1):
    diffX.append(float(Xs[i+1]) - float(Xs[i]) )
    diffY.append(float(Ys[i+1]) - float(Ys[i]) )
  #distance = np.sqrt(np.power(diffX,2) + np.power(diffY,2))
  #velocity = np.divide(distance,2) # 2ms gap!
  #velocity = np.absolute(velocity)
  Velocity = []
  direction=[]
  for i in range(len(diffX)):
    Velocity.append(diffX[i] + diffY[i])
    #direction.append(atan2(diffX[i], diffY[i]))
    velocity=np.divide(Velocity, t_dist)
    velocity=np.absolute(velocity)

  global mvmts 
  mvmts = []  
  #store 1 in mvmts[] if velocity is less than threshold else store 2
  for v in velocity:
    if(v<v_threshold):
        mvmts.append(0)
    else:
        mvmts.append(1)

  mvmts = enforce_min_duration(mvmts, min_fixation_duration_ms, ET_SAMPLING_RATE)

  # return mvmts,velocity
  return mvmts