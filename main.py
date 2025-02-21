import numpy as np
import pandas as pd
import os
import sys

from modules.dataloader import dataloader

# insert ACE-DNV submodules to be callable inside our code
inner_project_path = os.path.abspath("modules/methods/ACEDNV")
sys.path.insert(0, inner_project_path)

### Load Methods
from modules.methods.IVT import ivt
from modules.methods.IDT import idt
from modules.methods.gazeNet.myRun import pred as gazeNet
from modules.methods.remodnav.myRun import pred as remodnav 
from modules.methods.I2MC.I2MC_api import run as i2mc
from modules.methods.ACEDNV.modules.eventDetector import pred_detector as ACE_predictor
from modules.methods.ACEDNV.modules.reader import readDataset as aceReader

### Main body of execution

# loading the dataset
data = dataloader()

# TODO threshold to be optimized
# IVT algorithm execution
ivt_res = ivt(data[0], v_threshold=0.6)

# IDT algorithm execution
idt_res = idt(data[0], threshold=0.6)

# IM2C algorithm execution
i2mc_res = i2mc(data[0])


# gazeNet execution
data = np.array(data)

x_values = np.random.randint(0, 101, size=200)
y_values = np.random.randint(0, 101, size=200)
evt_values = np.random.randint(1, 4, size=200)
df = pd.DataFrame({
    'x': x_values,
    'y': y_values,
    'evt': evt_values
})
X_test_all = [df]
gazeNet_res = gazeNet(df)

# print(gazeNet_res)


# RemodNAV method
df = df.drop(['evt', 'status'], axis=1)
remo_res = remodnav(df)

# print(remo_res)

# ACE-DNV

ds_x, ds_y = aceReader()
ds_x = np.array(ds_x, dtype=object); 
if ds_y: ds_y = np.array(ds_y, dtype=object)

ace_res = ACE_predictor(ds_x, ds_y, "modules/methods/ACEDNV/model-zoo/random_forest.pkl")
print("done")


