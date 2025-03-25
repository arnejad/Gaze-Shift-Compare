# This script runs IVT and IDT algs to find the best-fitting threshold for them
# The used threhold in main.py are extracted based on executing this script
import numpy as np
from modules.methods.IVT import ivt
from modules.methods.IDT import idt
from modules.dataloader import dataloader
from modules.methods.ACEDNV.modules.scorer import score as scorer
from modules.utils import drawProgress
import matplotlib.pyplot as plt


plt.ion()


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

