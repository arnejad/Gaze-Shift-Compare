import numpy as np

from modules.dataloader import dataloader
from modules.methods.ACEDNV.modules.scorer import score as scorer
from modules.utils import ashScore
labeler = "EB"

_, labels_EB = dataloader(labeler, remove_blinks=False, degConv=False) # Note: Different methods have different dataloaders


labeler = "AG"

_, labels_AG = dataloader(labeler, remove_blinks=False, degConv=False) # Note: Different methods have different dataloaders


for i, labels in enumerate(labels_EB):
    rmidcs = np.where(labels == -1) # remove blinks
    labels_EB[i] = np.delete(labels_EB[i], rmidcs)
    labels_AG[i] = np.delete(labels_AG[i], rmidcs)

for i, labels in enumerate(labels_AG):
    rmidcs = np.where(labels == -1) # remove blinks
    labels_AG[i] = np.delete(labels_AG[i], rmidcs)
    labels_EB[i] = np.delete(labels_EB[i], rmidcs)


# for i, labels in enumerate(labels_EB):
#     for j, sample in enumerate(labels_EB[i]):
#         if labels_EB[i][j] != labels_AG[i][j]:
#             print("rec: " + str(i) + " sample: " + str(j))

f1s_m=[] #all f1 scores obtained in for this threshold on all recording
f1e_m=[]
ash_scores_m = []
for i, rec in enumerate(labels_AG):

    f1s_mi, f1e_mi = scorer(labels_EB[i], labels_AG[i], printBool=False)   #f1 scores for this recording on this threshold
    ash_score_mi = ashScore(labels_EB[i], labels_AG[i])
    f1s_m.append(f1s_mi)
    f1e_m.append(f1e_mi)
    ash_scores_m.append(ash_score_mi)

print("AG predict EB with:")
print("sample: " + str(np.mean(f1s_m)) + " event: " + str(np.mean(f1e_m)) + " ashscore: " + str(np.mean(ash_scores_m)))


f1s_m=[] #all f1 scores obtained in for this threshold on all recording
f1e_m=[]
ash_scores_m = []
for i, rec in enumerate(labels_AG):
    f1s_mi, f1e_mi = scorer(labels_AG[i], labels_EB[i], printBool=False)   #f1 scores for this recording on this threshold
    ash_score_mi = ashScore(labels_AG[i], labels_EB[i])
    f1s_m.append(f1s_mi)
    f1e_m.append(f1e_mi)
    ash_scores_m.append(ash_score_mi)

print("EB predict AG with:")
print("sample: " + str(np.mean(f1s_m)) + " event: " + str(np.mean(f1e_m)) + " ashscore: " + str(np.mean(ash_scores_m)))
