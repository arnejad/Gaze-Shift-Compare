
############################################################################################
# The code for this algorithm is extracted from 
# https://github.com/mebirtukan/Evaluation-of-eye-movement-event-detection-algorithms/blob/main/I_VT_Classifier.ipynb
# and redistributed under MIT license.
############################################################################################

import numpy as np

# we assume that the frequency is 500Hz so there is 2ms gap between every two samples
def ivt(data,v_threshold):
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
    velocity=np.divide(Velocity, 2)
    velocity=np.absolute(velocity)

  global mvmts 
  mvmts = []  
  #store 1 in mvmts[] if velocity is less than threshold else store 2
  for v in velocity:
    if(v<v_threshold):
        mvmts.append(0)
    else:
        mvmts.append(1)

  # return mvmts,velocity
  return mvmts