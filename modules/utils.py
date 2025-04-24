import numpy as np
import matplotlib.pyplot as plt
from modules.methods.ACEDNV.modules.scorer import score as scorer
from os import listdir
from os.path import isfile, join, isdir
from config import INP_DIR
import os
from sklearn.metrics import ConfusionMatrixDisplay


def drawProgress_justMean(sample_scores, event_scores, plotted_threshs, alg_name, fig=None, ax=None):

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.clear()
    ax.plot(plotted_threshs, sample_scores, marker='o', label='sample-based')
    ax.plot(plotted_threshs, event_scores, marker='s', label='event-based')
    ax.title("Score Progress")
    ax.xlabel("Iteration")
    ax.ylabel("Score")
    ax.legend()

    ax.draw()
    # Brief pause so the GUI event loop can update the figure
    ax.pause(0.1)

    return fig, ax


def drawProgress(sample_scores, event_scores, plotted_threshs, alg_name, fig=None, ax=None):

    sample_means = [np.mean(accs) for accs in sample_scores]
    event_means = [np.mean(accs) for accs in event_scores]

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.clear()

    for x, accs_list in zip(plotted_threshs, sample_scores):
        # We'll scatter them all vertically at x
        ax.scatter([x]*len(accs_list), accs_list, color='blue', alpha=0.5)

    for x, accs_list in zip(plotted_threshs, event_scores):
        # We'll scatter them all vertically at x
        ax.scatter([x]*len(accs_list), accs_list, color='red', alpha=0.5)

    ax.plot(plotted_threshs, sample_means, 'o-', color='blue', label='Sample-level')
    ax.plot(plotted_threshs, event_means, 'o-', color='red', label='Event-level')
    ax.set_title("Accuracies vs Threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1-Score")
    ax.legend()
    plt.draw()
    plt.pause(0.1)
    return fig, ax


def evaluate_old(methodFunc, data, labels):
    f1s_t=[] #all f1 scores obtained in for this threshold on all recording
    f1e_t=[]
    for i, rec in enumerate(data):
        print("rec: " + str(i))
        # Compute the score using the passed function
        preds = methodFunc(rec, v_threshold=1.2)
        f1s_ti, f1e_ti = scorer(preds, labels[i][:-1], printBool=False)   #f1 scores for this recording on this threshold
        f1s_t.append(f1s_ti)
        f1e_t.append(f1e_ti)


def confMat_visualizer(cm_s, cm_e, methodName):

    order = [1, 0]
    cm_s = cm_s[np.ix_(order, order)]
    cm_e = cm_e[np.ix_(order, order)]

    labels = ["Gaze-Shift", "Rest"]  # or whatever your classes are

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Sample-level confusion matrix
    disp_sample = ConfusionMatrixDisplay(confusion_matrix=cm_s, display_labels=labels)
    disp_sample.plot(ax=axes[0], cmap='Purples', values_format='.2f', colorbar=False)
    axes[0].set_title("Sample-level Confusion Matrix")

    # Event-level confusion matrix
    disp_event = ConfusionMatrixDisplay(confusion_matrix=cm_e, display_labels=labels)
    disp_event.plot(ax=axes[1], cmap='Greens', values_format='.2f', colorbar=False)
    axes[1].set_title("Event-level Confusion Matrix")

    for text in disp_sample.text_.ravel():
        text.set_fontsize(20)

    for text in disp_event.text_.ravel():
        text.set_fontsize(20)

    plt.tight_layout()
    plt.savefig(methodName+'_confMats.png')

# This function recievs list of methods and their input parameters and run them
# one by one on each data recording.
def evaluate(methodList, data, labels):
    
    
    f1s_all = []
    f1e_all = []
    ash_scores_all = []
    for method, params in methodList:
        print("method: " + method.__name__)
        f1s_m=[] #all f1 scores obtained in for this threshold on all recording
        f1e_m=[]
        ash_scores_m = []
        cm_s_all = [[0,0],[0,0]]
        cm_e_all = [[0,0],[0,0]]
        for i, rec in enumerate(data):
            print("rec: " + str(i))
            # Compute the score using the passed function
            if method.__name__ == "ivt":
                adjusted_labels = labels[i][:-1]
            if method.__name__ == "idt":
                adjusted_labels = labels[i][1:-1]        
            # if method.__name__ != "ranking" and method.__name__ != "runMovingWindow" and method.__name__ != "runOEMC": 
            if method.__name__ not in ["ranking", "runMovingWindow", "runOEMC"]:
                preds = method(rec, **params)
            else: preds = rec

            if method.__name__ == "gazeNet":
                adjusted_labels = preds[1] 
                preds = preds[0]
            if method.__name__ in {"ACEDNV", "ranking", "runMovingWindow", "runOEMC"}:
                adjusted_labels = labels[i]
            if method.__name__ == "ACEDNV":
                preds = np.array(preds)
            
            f1s_mi, f1e_mi, cm_s, cm_e = scorer(preds, adjusted_labels, printBool=False)   #f1 scores for this recording on this threshold
            # ash_score_mi = ashScore(preds, adjusted_labels)
            f1s_m.append(f1s_mi)
            f1e_m.append(f1e_mi)
            # ash_scores_m.append(ash_score_mi)
            cm_s_all = cm_s_all+cm_s
            cm_e_all = cm_e_all+cm_e
        f1s_all.append(f1s_m)
        f1e_all.append(f1e_m)
        cm_s_avg = np.array(cm_s_all)/(i+1)
        cm_e_avg = np.array(cm_e_all)/(i+1)
        # ash_scores_all.append(ash_scores_m)

    # return f1s_all, f1e_all, ash_scores_all, cm_s_avg, cm_e_avg
    print("Method: " + method.__name__)
    print("sample: " + str(np.mean(f1s_all)) + " event: " + str(np.mean(f1e_all)) + " ashscore: " + str(np.mean(ash_scores_all)))
    confMat_visualizer(cm_s_avg, cm_e_avg, method.__name__)
        



def getRanges(arr):
    diff = np.diff(arr, prepend=0, append=0) # Find indices where value changes
    start_indices = np.where(diff == 1)[0] # Start indices of 1s (where diff == 1)
    end_indices = np.where(diff == -1)[0] - 1    # End indices of 1s (where diff == -1) - 1 to get the correct end index
    chunks = np.column_stack((start_indices, end_indices)) # Combine start and end indices into tuples
    return chunks



def ashScore(pred, gt):
    
    # get the gt->pred matching score
    ranges = getRanges(gt)
    innerScore = 0
    for event in ranges:
        TP_i = np.sum(pred[event[0]:event[1]+1])
        n_i = event[1] - event[0] + 1
        innerScore += TP_i/n_i

    GT_match_score = innerScore/(2*ranges.shape[0])

    # get pred->gt matching score
    ranges = getRanges(pred)
    innerScore = 0
    for event in ranges:
        TP_i = np.sum(gt[event[0]:event[1]+1])
        n_i = event[1] - event[0] + 1
        innerScore += TP_i/n_i

    Pred_match_score = innerScore/(2*ranges.shape[0])

    return Pred_match_score + GT_match_score

    print(ranges)




def cacheLoadedData(data):
    
    recs = [f for f in listdir(INP_DIR) if isdir(join(INP_DIR, f))]
    # recs = ['p51', 'p52']     #uncomment while running optimization
    for idx, array in enumerate(data):
        file_path = os.path.join("degs_cached", recs[idx]+".csv")
        np.savetxt(file_path, array, delimiter=",", fmt="%.5f")  # Save with 5 decimal precision
        print(f"Saved: {file_path}")