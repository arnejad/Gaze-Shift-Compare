import numpy as np
import os
import sys

inner_project_path = os.path.abspath("modules/methods/OEMC")
sys.path.insert(0, inner_project_path)


from modules.methods.OEMC.online_sim import OnlineSimulator as OEMC_OnlineSimulator
from modules.methods.OEMC.argsProducer import produceArgs as OEMC_ArgsReplicator
from modules.methods.OEMC.preprocessor import Preprocessor as oemc_preprocessor
from modules.methods.ACEDNV.modules.scorer import score



from config import INP_DIR



oemc_args = OEMC_ArgsReplicator()
oemc_pproc = oemc_preprocessor(window_length=1,offset=oemc_args.offset,
                                      stride=oemc_args.strides,frequency=250)
oemc_pproc.process_folder(INP_DIR, 'cached/VU')
oemcSimulator = OEMC_OnlineSimulator(oemc_args)
preds, gt = oemcSimulator.simulate(1)
f1_s, f1_e = score(preds, gt, printBool=False)
print("sample: " + str(np.mean(f1s)) + " event: " + str(np.mean(f1e)) + " ashscore: " + str(np.mean(ashscore)))


# from modules.dataloader import dataloader, converDataToGazeNet
# from modules.utils import evaluate
# from modules.methods.gazeNet.myRun import gazeNet
# data, labels = dataloader(remove_blinks=False, degConv=True)
# df = converDataToGazeNet(data, labels, dummy=False)
# f1s, f1e, ashscore = evaluate([(gazeNet, {})], df, labels)
# print("sample: " + str(np.mean(f1s)) + " event: " + str(np.mean(f1e)) + " ashscore: " + str(np.mean(ashscore)))