import numpy as np

from modules.dataloader import dataloader

### Load Methods
from modules.methods.IVT import ivt
from modules.methods.IDT import idt

### Main body of execution

# loading the dataset
data = dataloader()


# TODO threshold to be optimized
# IVT algorithm execution
ivt_res = ivt(data[0], v_threshold=0.6)

# IDT algorithm execution
idt_res = idt(data[0], threshold=0.6)

