import numpy as np
from modules.methods.IVT import ivt
from modules.methods.IDT import idt
from modules.dataloader import dataloader
from modules.methods.ACEDNV.modules.scorer import score as scorer
import matplotlib.pyplot as plt


plt.ion()


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


def optimize_threshold(
    func_to_optimize,  # the function whose score you want to optimize
    threshold_range,   # iterable of possible thresholds
    data, labels, alg_name
):
    fig, ax = None, None  # we'll keep track of these so we update the same figure each time
    best_threshold = None
    best_score = None  # might be float('inf') or -inf depending on best direction
    f1s_all = []    #f1 scores of each threshold for each recording
    f1e_all = []
    for t, threshold in enumerate(threshold_range):
        print("Threshold: " + str(threshold))
        f1s_t=[] #all f1 scores obtained in for this threshold on all recording
        f1e_t=[]
        for i, rec in enumerate(data):
            print("rec: " + str(i))
            # Compute the score using the passed function
            preds = func_to_optimize(rec, threshold)
            f1s_ti, f1e_ti = scorer(preds, labels[i], printBool=False)   #f1 scores for this recording on this threshold
            f1s_t.append(f1s_ti)
            f1e_t.append(f1e_ti)

        f1s_all.append(f1s_t)
        f1e_all.append(f1e_t)
        fig, ax = drawProgress(f1s_all, f1e_all, threshold_range[:t+1],  alg_name, fig, ax)
        if best_score is None or np.mean(f1e_t) > best_score:
            best_threshold = threshold
            best_score = np.mean(f1e_t)
    plt.ioff()
    plt.savefig(alg_name+".png", dpi=300)
    plt.show(block=False)

    return best_threshold, best_score



data, labels = dataloader(remove_blinks=True, degConv=False)

# best_thr_2, best_score_2 = optimize_threshold(idt, np.arange(0, 60, 1), data, [sub_list[1:-1] for sub_list in labels], "idt")
# print("idt 2 best:", best_thr_2, best_score_2)


best_thr_1, best_score_1 = optimize_threshold(ivt, np.arange(0, 7, 0.5), data, [sub_list[:-1] for sub_list in labels], "ivt")
# best_thr_1, best_score_1 = optimize_threshold(ivt, np.arange(1, 2, 0.5), data, [sub_list[:-1] for sub_list in labels], "ivt")

print("ivt 1 best:", best_thr_1, best_score_1)

