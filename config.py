# INP_DIR = "/home/ash/projects/video_for_Ashkan/exports"
INP_DIR = '/media/ash/Expansion/data/Saccade-Detection-Methods'
DATASET = "PI"              #choose self collected data from Pupil Ivisible by "PI", and gaze-in-wild dataset by "GiW"

CLOUD_FORMAT = False
VIDEO_SIZE = [1080, 1088]   # [width, height]

ET_SAMPLING_RATE = 250      # Sampling rate of the eye-movements in Hz

OEMC_MODEL = "modules/methods/OEMC/final_models/tcn_model_hmr_BATCH-2048_EPOCHS-25_FOLD-5.pt"

MATLAB_PATH = "/home/ash/MATLAB/bin/matlab"