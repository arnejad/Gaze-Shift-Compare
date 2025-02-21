# This code has been developed by Ashkan Nejad at Royal Dutch Visio and Univeristy Medical Center Groningen
# personal conda env: dfvo2

import numpy as np

from modules.eventDetector import eventDetector_new as train_RF
from modules.eventDetector import pred_detector as pred_RF
from modules.preprocess import data_balancer
from modules.reader import readDataset

########## DATA PREPARATION

ds_x, ds_y = readDataset()

# Preprocess

# ds_x, ds_y = data_balancer(ds_x, ds_y)


# f1s_sample = []
# f1s_event = []
# for p in range(1, len(ds_y)):

#     x_test = ds_x[p]
#     y_test =  ds_y[p]
#     x_train = np.array(ds_x)
#     x_train = np.delete(ds_x, p, 0)
#     y_train = np.array(ds_y)
#     y_train = np.delete(ds_y, p, 0)
#     f1_ie, f1_is = NEED_TrainAndTest(x_train, y_train, x_test, y_test)
#     f1s_sample.append(f1_ie)
#     f1s_event.append(f1_is)

# feats,lbls = preprocessor(gazes, patchDists, headRot, T, TMatch, labels, lblMatch)
ds_x = np.array(ds_x, dtype=object); 

if ds_y: ds_y = np.array(ds_y, dtype=object)



pred_RF(ds_x, ds_y, "model-zoo/random_forest.pkl")

# data_stats(ds_y)


# plt.hist(ds_y, bins=np.arange(6))
# plt.show()

train_RF(ds_x, ds_y)

# for prediction using trained model
# preds = pred_RF(ds_x, ds_y, OUT_DIR+'/models/RF_lblr6.sav')

