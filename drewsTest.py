from modules.dataloader import dataloader
from modules.utils import evaluate
from modules.methods.IVT import ivt
from modules.methods.IDT import idt
from modules.methods.Hooge.run import runMovingWindow

from config import INP_DIR

data, labels = dataloader("DrewsDynamic", INP_DIR, None, remove_blinks=True, degConv=False, incTimes=True)
preds = runMovingWindow(data, 6000, 2.7)
evaluate([(runMovingWindow, {})], preds, labels)


data, labels = dataloader("DrewsDynamic", INP_DIR, None, remove_blinks=True, degConv=False)

methods = [
    (ivt, {"v_threshold": 2.0, "min_fixation_duration_ms":15}),
    (idt, {"d_threshold": 26,  "min_duration_ms":15})
]
evaluate(methods, data, labels)

