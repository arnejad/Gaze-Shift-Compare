import numpy as np

from modules.dataloader import dataloader
from modules.methods.IVT import ivt

# loading the dataset
data = dataloader()


# IVT algorithm execution, TODO threshold to be optimized
ivt_res = ivt(data[0], v_threshold=0.6)


