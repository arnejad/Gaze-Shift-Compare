# This script runs IVT and IDT algs to find the best-fitting threshold for them
# The used threhold in main.py are extracted based on executing this script
import numpy as np
from modules.methods.IVT import ivt
from modules.methods.IDT import idt
from modules.dataloader import dataloader
from modules.methods.ACEDNV.modules.scorer import score as scorer
from modules.methods.Hooge.run import runMovingWindow
from modules.utils import drawProgress
import matplotlib.pyplot as plt


plt.ion()

#optimize on one sample from both labelers
singleSampleBothLabler = True
method="ivt" #choose between "ivt", "idt" and "mw"

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

def showHeatmap(scoreTable, thresholds1, thresholds2, mode, alg_name):
    fig, ax = plt.subplots()
    c = ax.imshow(scoreTable, cmap='viridis', origin='lower', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(thresholds2)))
    ax.set_yticks(np.arange(len(thresholds1)))
    ax.set_xticklabels(thresholds2)
    ax.set_yticklabels(thresholds1)

    if alg_name == "mw":
        ax.set_xlabel('Lambda')
        ax.set_ylabel('Window size (samples)')
        alg_name_tp = "Moving Window"
    elif alg_name == "idt":
        ax.set_ylabel('Dispersion threshold (pixels)')
        ax.set_xlabel('Minimum duration (ms)')
        alg_name_tp = "I-DT"
    elif alg_name == "ivt":
        ax.set_ylabel('Velocity threshold (pixels/ms)')
        ax.set_xlabel('Minimum duration (ms)')
        alg_name_tp = "I-VT"

    ax.set_title(alg_name_tp + ' Performance Heatmap')
    fig.colorbar(c, ax=ax, label='Performance Score (%)')

    for i in range(len(thresholds1)):
        for j in range(len(thresholds2)):
            value = scoreTable[i, j]
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', color='white')

    plt.tight_layout()
    plt.ioff()
    plt.savefig(alg_name_tp+"-"+mode+".png", dpi=300)
    plt.show(block=False)


def optimizeDoubleThreshold(func_to_optimize,  # the function whose score you want to optimize
    thresholds1, thresholds2, data, labels, alg_name):
    f1s_all = np.zeros((len(thresholds1), len(thresholds2)))    #f1 scores of each threshold for each recording
    f1e_all = np.zeros((len(thresholds1), len(thresholds2)))

    for t1_i, t1 in enumerate(thresholds1):
        for t2_i, t2 in enumerate(thresholds2):
            print("Thresholds: " + str(t1) + " and " + str(t2))
            f1s_t=[] #all f1 scores obtained in for this threshold on all recording
            f1e_t=[]

            # Compute the score using the passed function

            for i, rec in enumerate(data):
                print("rec: " + str(i))
                preds = func_to_optimize(rec, t1, t2)
                f1s_ti, f1e_ti, _,_ = scorer(preds, labels[i], printBool=False)   #f1 scores for this recording on this threshold
                f1s_t.append(f1s_ti)
                f1e_t.append(f1e_ti)
                
            f1s_all[t1_i, t2_i] = np.mean(f1s_t)
            f1e_all[t1_i, t2_i] = np.mean(f1e_t)

    showHeatmap(f1e_all, thresholds1, thresholds2, "event", alg_name)
    showHeatmap(f1s_all, thresholds1, thresholds2, "sample", alg_name)

    print("Parameter check complete")


if singleSampleBothLabler:

    if method=="mw":
        data_EB, labels_EB = dataloader("EB", remove_blinks=True, degConv=True, incTimes=True)
        data_EB, labels_EB = dataloader("EB", remove_blinks=True, degConv=True, incTimes=True)

        data_AG, labels_AG = dataloader("AG", remove_blinks=True, degConv=True, incTimes=True)
        data_AG, labels_AG = dataloader("AG", remove_blinks=True, degConv=True, incTimes=True)
    else:
        data_EB, labels_EB = dataloader("EB", remove_blinks=True, degConv=False, incTimes=False)
        data_EB, labels_EB = dataloader("EB", remove_blinks=True, degConv=False, incTimes=False)

        data_AG, labels_AG = dataloader("AG", remove_blinks=True, degConv=False, incTimes=False)
        data_AG, labels_AG = dataloader("AG", remove_blinks=True, degConv=False, incTimes=False)
        
    data = [data_EB[0], data_AG[0]]
    labels = [labels_EB[0], labels_AG[0]]
else:
    data, labels = dataloader("EB", remove_blinks=True, degConv=False)


if method=="idt":
    # best_thr_2, best_score_2 = optimize_threshold(idt, np.arange(0, 60, 1), data, [sub_list[1:-1] for sub_list in labels], "idt")
    optimizeDoubleThreshold(idt, np.arange(10, 26, 1), np.arange(0, 101, 25), data, [sub_list[1:-1] for sub_list in labels], "idt")
    # print("idt 2 best:", best_thr_2, best_score_2)

elif method=="ivt":
    # best_thr_1, best_score_1 = optimize_threshold(ivt, np.arange(0, 7, 0.5), data, [sub_list[:-1] for sub_list in labels], "ivt")
    # best_thr_1, best_score_1 = optimize_threshold(ivt, np.arange(0, 3.5, 0.5), data, [sub_list[:-1] for sub_list in labels], "ivt")
    optimizeDoubleThreshold(ivt, np.arange(0, 3.5, 0.5), np.arange(0, 101, 25), data, [sub_list[:-1] for sub_list in labels], "ivt")

    # print("ivt 1 best:", best_thr_1, best_score_1)
elif method=="mw":
    print("s")
    optimizeDoubleThreshold(runMovingWindow, np.arange(3000, 10000, 1000), np.arange(1, 5.5, 0.5), data, labels)
    # optimizeDoubleThreshold(runSlidingWindow, np.arange(8000, 9001, 1000), np.arange(2.5, 3.5, 0.5), data, labels)