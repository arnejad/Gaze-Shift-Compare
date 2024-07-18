############################################################################################
# The code for this algorithm is extracted from https://github.com/dcnieho/I2MC_Python 
# and redistributed under MIT license.
############################################################################################
import os
import sys
import pandas as pd
import numpy as np
import modules.methods.I2MC.I2MC as I2MC
from config import ET_SAMPLING_RATE


opt = {}
# General variables for eye-tracking data
opt['xres']         = 1920.0                # maximum value of horizontal resolution in pixels
opt['yres']         = 1080.0                # maximum value of vertical resolution in pixels
opt['missingx']     = -opt['xres']          # missing value for horizontal position in eye-tracking data (example data uses -xres). used throughout the algorithm as signal for data loss
opt['missingy']     = -opt['yres']          # missing value for vertical position in eye-tracking data (example data uses -yres). used throughout algorithm as signal for data loss
opt['freq']         = float(ET_SAMPLING_RATE)# sampling frequency of data (check that this value matches with values actually obtained from measurement!)

# Variables for the calculation of angular measures
# These values are used to calculate noise measures (RMS and BCEA) of
# fixations. The may be left as is, but don't use the noise measures then.
# If either or both are empty, the noise measures are provided in pixels
# instead of degrees.
opt['scrSz']        = [50.9174, 28.6411]    # screen size in cm
opt['disttoscreen'] = 65.0                  # distance to screen in cm.



def run(data):
    print("good")

