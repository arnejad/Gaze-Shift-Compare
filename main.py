import numpy as np
import pandas as pd
from modules.dataloader import dataloader

### Load Methods
from modules.methods.IVT import ivt
from modules.methods.IDT import idt
from modules.methods.gazeNet.myRun import pred as gazeNet
from modules.methods import remodnav
### Main body of execution

remodnav.main()

# loading the dataset
data = dataloader()


# TODO threshold to be optimized
# IVT algorithm execution
ivt_res = ivt(data[0], v_threshold=0.6)

# IDT algorithm execution
idt_res = idt(data[0], threshold=0.6)


# RemodNAV method



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

print(gazeNet_res)
