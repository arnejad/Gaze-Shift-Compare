import numpy as np

from modules.dataloader import dataloader, converDataToGazeNet
from modules.utils import evaluate
from modules.methods.gazeNet.myRun import gazeNet


data, labels = dataloader(remove_blinks=False, degConv=True)
df = converDataToGazeNet(data, labels, dummy=False)
f1s, f1e, ashscore = evaluate([(gazeNet, {})], df, labels)
print("sample: " + str(np.mean(f1s)) + " event: " + str(np.mean(f1e)) + " ashscore: " + str(np.mean(ashscore)))