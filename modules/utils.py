import numpy as np
import matplotlib.pyplot as plt
from modules.methods.ACEDNV.modules.scorer import score as scorer
from os import listdir
from os.path import isfile, join, isdir
from config import INP_DIR
import os
from sklearn.metrics import ConfusionMatrixDisplay
import glob


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
    
    
    for method, params in methodList:
        print("method: " + method.__name__)
        f1s_m=[] #all f1 scores obtained in for this threshold on all recording
        f1e_m=[]
        ash_scores_m = []
        cm_s_m = [[0,0],[0,0]]
        cm_e_m = [[0,0],[0,0]]
        for i, rec in enumerate(data):
            print("rec: " + str(i))
            # Compute the score using the passed function
            if method.__name__ == "ivt":
                adjusted_labels = labels[i][:-1]
            if method.__name__ == "idt":
                adjusted_labels = labels[i][1:-1]        
            # if method.__name__ != "ranking" and method.__name__ != "runMovingWindow" and method.__name__ != "runOEMC": 
            if method.__name__ not in ["runRankingMethod", "runMovingWindow", "runOEMC"]:
                preds = method(rec, **params)
            else: preds = rec

            if method.__name__ == "gazeNet":
                adjusted_labels = preds[1] 
                preds = preds[0]
            if method.__name__ in {"ACEDNV", "runRankingMethod", "runMovingWindow", "runOEMC"}:
                adjusted_labels = labels[i]
            if method.__name__ == "ACEDNV":
                preds = np.array(preds)
            
            f1s_mi, f1e_mi, cm_s, cm_e = scorer(preds, adjusted_labels, printBool=False)   #f1 scores for this recording on this threshold
            # ash_score_mi = ashScore(preds, adjusted_labels)
            f1s_m.append(f1s_mi)
            f1e_m.append(f1e_mi)
            # ash_scores_m.append(ash_score_mi)
            cm_s_m = cm_s_m+cm_s
            cm_e_m = cm_e_m+cm_e

        cm_s_m_avg = np.array(cm_s_m)/(i+1)
        cm_e_m_avg = np.array(cm_e_m)/(i+1)
        outputPerformance(method.__name__, f1s_m, f1e_m, cm_s_m_avg, cm_e_m_avg)
        # ash_scores_all.append(ash_scores_m)

    # return f1s_all, f1e_all, ash_scores_all, cm_s_avg, cm_e_avg
    
    # print("Method: " + method.__name__)
    # print("sample: " + str(np.mean(f1s_all)) + " event: " + str(np.mean(f1e_all)) + " ashscore: " + str(np.mean(ash_scores_all)))
    # confMat_visualizer(cm_s_avg, cm_e_avg, method.__name__)
        
def outputPerformance(methodName, allF1s, allF1e, cm_s_avg, cm_e_avg):
    print("Method: " + methodName)
    print("sample: " + str(np.mean(allF1s)) + " event: " + str(np.mean(allF1e)))
    confMat_visualizer(cm_s_avg, cm_e_avg, methodName)


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


def clearCache(dir):
    files = glob.glob(os.path.join(dir, "*"))
    for f in files:
        if os.path.isfile(f):   # only delete files, not subfolders
            os.remove(f)

def cacheLoadedData(data):
    
    recs = [f for f in listdir(INP_DIR) if isdir(join(INP_DIR, f))]

    if len(data) > 30:
        data = [data]
        recs = ['temp']
    # recs = ['p51', 'p52']     #uncomment while running optimization
    clearCache("degs_cached")
    for idx, array in enumerate(data):
        file_path = os.path.join("degs_cached", recs[idx]+".csv")
        np.savetxt(file_path, array, delimiter=",", fmt="%.5f")  # Save with 5 decimal precision
        print(f"Saved: {file_path}")



def longSaccadeRemover(timestamps, labels, threshold_ms=150.0):
    """
    Turn to 0 any contiguous segment of 1's in `labels` whose duration
    (based on `timestamps`) is strictly greater than `threshold_ms`.

    Parameters
    ----------
    timestamps : array-like of float
        1D, strictly increasing time stamps (in seconds; negative start OK).
    labels : array-like of int/bool
        1D labels aligned with timestamps, containing 0/1.
    threshold_ms : float
        Duration threshold in milliseconds.

    Returns
    -------
    np.ndarray
        A copy of labels with long 1-segments flipped to 0.
    """
    t = np.asarray(timestamps, dtype=float)
    y = np.asarray(labels, dtype=int).copy()

    if t.ndim != 1 or y.ndim != 1 or t.shape[0] != y.shape[0]:
        raise ValueError("timestamps and labels must be 1D and of equal length")
    if np.any(np.diff(t) <= 0):
        raise ValueError("timestamps must be strictly increasing")

    thr_s = threshold_ms / 1000.0
    n = len(y)
    i = 0

    while i < n:
        if y[i] == 1:
            start = i
            # walk to end of this 1-run
            while i + 1 < n and y[i + 1] == 1:
                i += 1
            end = i

            # Duration estimate: last sample time minus first sample time.
            # (This assumes per-sample labeling; adjust if you use interval labels.)
            duration = t[end] - t[start]

            if duration > thr_s:          # “longer than” the threshold
                y[start:end + 1] = 0

        i += 1

    return y


def find_long_saccade_segments(timestamps, labels, threshold_ms=100.0):
    """
    Find indices of contiguous 1-segments longer than threshold.

    Parameters
    ----------
    timestamps : array-like of float
        1D strictly increasing timestamps (seconds).
    labels : array-like of int/bool
        1D labels aligned with timestamps (0/1).
    threshold_ms : float
        Duration threshold in milliseconds.

    Returns
    -------
    indices_to_remove : np.ndarray
        Indices of labels that belong to '1' segments longer than threshold.
    """
    t = np.asarray(timestamps, dtype=float)
    y = np.asarray(labels, dtype=int)

    if t.ndim != 1 or y.ndim != 1 or t.shape[0] != y.shape[0]:
        raise ValueError("timestamps and labels must be 1D and of equal length")
    if np.any(np.diff(t) <= 0):
        raise ValueError("timestamps must be strictly increasing")

    thr_s = threshold_ms / 1000.0
    n = len(y)
    indices_to_remove = []

    i = 0
    while i < n:
        if y[i] == 1:
            start = i
            while i + 1 < n and y[i + 1] == 1:
                i += 1
            end = i
            duration = t[end] - t[start]
            if duration > thr_s:
                indices_to_remove.extend(range(start, end + 1))
        i += 1

    return np.array(indices_to_remove, dtype=int)


def plot_one_segment_durations_across_recordings(timestamps_list, labels_list, threshold_ms=None):
    """
    Compute and plot the distribution of durations (ms) of all contiguous 1-segments
    across multiple recordings.

    Parameters
    ----------
    timestamps_list : Sequence[Sequence[float]]
        2D-like structure: each element is a 1D strictly-increasing timestamp array (seconds).
    labels_list : Sequence[Sequence[int or bool]]
        2D-like structure: each element is a 1D label array (0/1) aligned with the
        timestamps array at the same index.
    threshold_ms : float or None, optional
        If provided, draw a vertical line at this threshold on the histogram (ms).

    Returns
    -------
    np.ndarray
        Array of all segment durations (ms) found across all recordings (useful for stats).
    """
    if len(timestamps_list) != len(labels_list):
        raise ValueError("timestamps_list and labels_list must have the same number of recordings")

    all_durations_ms = []

    for rec_idx, (t_arr, y_arr) in enumerate(zip(timestamps_list, labels_list)):
        t = np.asarray(t_arr, dtype=float)
        y = np.asarray(y_arr, dtype=int)

        if t.ndim != 1 or y.ndim != 1:
            raise ValueError(f"Recording {rec_idx}: timestamps and labels must be 1D arrays")
        if t.shape[0] != y.shape[0]:
            raise ValueError(f"Recording {rec_idx}: timestamps and labels length mismatch")
        if np.any(np.diff(t) <= 0):
            raise ValueError(f"Recording {rec_idx}: timestamps must be strictly increasing")

        n = len(y)
        i = 0
        while i < n:
            if y[i] == 1:
                start = i
                while i + 1 < n and y[i + 1] == 1:
                    i += 1
                end = i
                duration_ms = (t[end] - t[start]) * 1000.0
                all_durations_ms.append(duration_ms)
            i += 1

    # --- Plot combined distribution ---
    plt.figure()
    if len(all_durations_ms) > 0:
        plt.hist(all_durations_ms, bins="auto", edgecolor="black", alpha=0.85)
        if threshold_ms is not None:
            plt.axvline(threshold_ms, linestyle="--", linewidth=2)
        plt.title("Distribution of gaze shift durations across recordings")
        plt.xlabel("Duration (ms)")
        plt.ylabel("Count")
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, "No segments with label=1 found in any recording",
                 ha="center", va="center", transform=plt.gca().transAxes)
        plt.axis("off")
        plt.title("Distribution of label=1 segment durations across recordings")
        plt.tight_layout()
    
    plt.savefig('distribution', dpi=300)
    plt.close()
    return np.array(all_durations_ms, dtype=float)