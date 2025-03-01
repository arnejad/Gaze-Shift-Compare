import numpy as np
import matplotlib.pyplot as plt
from modules.methods.ACEDNV.modules.scorer import score as scorer

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
    ax.set_title("Accuracies vs Threshold (Multiple Measurements)")
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

# This function recievs list of methods and their input parameters and run them
# one by one on each data recording.
def evaluate(methodList, data, labels):
    
    f1s_all = []
    f1e_all = []
    for method, params in methodList:
        print("method: " + method.__name__)
        f1s_m=[] #all f1 scores obtained in for this threshold on all recording
        f1e_m=[]
        for i, rec in enumerate(data):
            print("rec: " + str(i))
            # Compute the score using the passed function
            if method.__name__ == "ivt":
                adjusted_labels = labels[i][:-1]
            if method.__name__ == "idt":
                adjusted_labels = labels[i][1:-1] 
            
            preds = method(rec, **params)
           
            if method.__name__ == "gazeNet":
                adjusted_labels = preds[1] 
                preds = preds[0]
            if method.__name__ == "ACEDNV":
                adjusted_labels = labels[i]
            f1s_mi, f1e_mi = scorer(preds, adjusted_labels, printBool=False)   #f1 scores for this recording on this threshold
            f1s_m.append(f1s_mi)
            f1e_m.append(f1e_mi)
        f1s_all.append(f1s_m)
        f1e_all.append(f1e_m)

    return f1s_all, f1e_all
        