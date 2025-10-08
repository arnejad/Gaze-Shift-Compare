import os
import numpy as np

# root directory where all your folders are
root_dir = "/media/ash/Expansion/data/drews-dynamic-removed"

for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    if os.path.isdir(folder_path):  # only process folders
        file_path = os.path.join(folder_path, "gt_labels.npy")
        if os.path.exists(file_path):
            # load
            arr = np.load(file_path)

            # toggle 0 ↔ 1
            toggled = 1 - arr

            # save beside it
            save_path = os.path.join(folder_path, "gt_labels_toggled.npy")
            np.save(save_path, toggled)

            print(f"Toggled and saved: {save_path}")