import numpy as np

from modules.methods.OEMC.online_sim import OnlineSimulator as OEMC_OnlineSimulator
from modules.methods.OEMC.argsProducer import produceArgs as OEMC_ArgsReplicator
from modules.methods.OEMC.preprocessor import Preprocessor as oemc_preprocessor
from modules.dataloader import listRecNames
from modules.methods.ACEDNV.modules.scorer import score

from config import INP_DIR, LABELER



def runOEMC():
    print("Running OEMC:")
    oemc_args = OEMC_ArgsReplicator()
    oemc_pproc = oemc_preprocessor(window_length=1,offset=oemc_args.offset,
                                        stride=oemc_args.strides,frequency=250)
    oemc_pproc.process_folder(INP_DIR, 'cached/VU', LABELER)
    oemcSimulator = OEMC_OnlineSimulator(oemc_args)
    recs = listRecNames()
    # recs = ['p5'] # for debugging
    preds_all = []
    gt_all = []
    for r in recs:
        print("Recording: " + r + "\n")
        preds, gt = oemcSimulator.simulate(r, 1)

        preds = np.array(preds)
        gt = np.array(gt)

        preds[np.where(preds != 1)] = 0 #rest of the event (except saccade) mark as label 0
        
        rmidcs = np.where(gt == 3) # remove blinks
        gt = np.delete(gt, rmidcs)
        preds = np.delete(preds, rmidcs)

        preds_all.append(preds)
        gt_all.append(gt)
        
    return preds_all, gt_all

    f1_s, f1_e, ashscore = score(preds, gt, printBool=False)
