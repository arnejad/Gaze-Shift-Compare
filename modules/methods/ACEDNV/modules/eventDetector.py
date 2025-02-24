import numpy as np
import modules.visualizer as visual
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from modules.decisionMaker import print_scores as print_scores
from sklearn.model_selection import train_test_split
from modules.preprocess import data_stats, divider
from modules.scorer import score
import matplotlib.pyplot as plt
import torch
import pickle
import joblib

from modules.methods.ACEDNV.config import PATCH_SIM_THRESH, GAZE_DIST_THRESH, ENV_CHANGE_THRESH, PATCH_SIZE, LAMBDA, PATCH_PRIOR_STEPS, OUT_DIR


def validate(model_dir, feats, lbls):
    clf = RandomForestClassifier(random_state=0, criterion='gini', n_estimators=300, max_features= 'log2', min_samples_leaf=1, max_depth=50, min_samples_split=2, bootstrap=False)
    with open(model_dir, 'rb') as f:
        clf = pickle.load(f)

    preds = clf.predict(feats)

    return preds



def eventDetector_new(feats, lbls):


    # # chunk random division
    # X_train = []
    # Y_train = []
    # X_test = []
    # Y_test = []
    # for i in range(feats.size):
    #     randStart = randrange(int(0.8*rec.size))
    #     rec = feats[i]
    #     lbl = lbls[i]
    #     recLen = int(0.2*rec.size)

    #     # features
    #     X_test.append(rec[randStart: (randStart+recLen)]) # add 20% lenght chuck to test
    #     rec = np.delete(rec, np.arange(randStart, (randStart+recLen)))  # delete the test chunk
    #     X_train.append(rec) # append the 80% rest for the training the train set

    #     # labels
    #     Y_test.append(lbl[randStart: (randStart+recLen)]) # add 20% lenght chuck to test
    #     rec = np.delete(lbl, np.arange(randStart, (randStart+recLen)))  # delete the test chunk
    #     Y_train.append(lbl) # append the 80% rest for the training the train set


    # feats = np.squeeze(feats)
    # feats = np.concatenate(feats)

    # lbls = np.squeeze(lbls)
    # lbls = np.concatenate(lbls)

    #sample-level random split
    # X_train, X_test, y_train, y_test = train_test_split(feats, lbls, test_size=0.2, random_state=42)
    
    X_train, y_train, X_test, y_test = divider(feats,lbls)
    X_train, ekkh, y_train, ekhg = train_test_split(X_train, y_train, test_size=1, random_state=42)

    
    data_stats(lbls)

    clf = RandomForestClassifier(random_state=0, criterion='gini', n_estimators=300, max_features= 'log2', min_samples_leaf=1, max_depth=50, min_samples_split=2, bootstrap=False)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

   
    f1_e, f1_s = score(preds, y_test)

    return f1_e, f1_s




def pred_detector(feats, lbls, modelDir):

    # if len([feats.size]) > 1:
    feats = np.concatenate(feats)
    feats = np.squeeze(feats)
    
   
    lbls = np.squeeze(lbls)
    lbls = np.concatenate(lbls)


    # clf = pickle.load(open(modelDir, 'rb'))
    # with open('model-zoo/rf_model', 'rb') as f:
    #     clf = pickle.load(f, encoding='latin1')

    clf = joblib.load(modelDir)

    preds = clf.predict(feats)

    
    # if lbls: 
    preds = torch.from_numpy(preds)
    lbls = torch.from_numpy(lbls)
        

    # if lbls: 
    # print_scores(preds, lbls, 0, 'RF')
    score(preds, lbls)

    return preds


    
def trainAndTest(x_train, y_train, x_test, y_test):
    x_train = np.squeeze(x_train)
    x_train = np.concatenate(x_train)

    y_train = np.squeeze(y_train)
    y_train = np.concatenate(y_train)

    x_test = np.squeeze(x_test)

    y_test = np.squeeze(y_test)


    clf = RandomForestClassifier(random_state=0, criterion='gini', n_estimators=300, max_features= 'log2', min_samples_leaf=1, max_depth=50, min_samples_split=2, bootstrap=False)
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)

    # f1_e, f1_s = score(preds, y_test)

    return preds


