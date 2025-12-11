# Instruction on how to use Ranking Method gaze-shift/saccade detection

This document provides a clean, professional overview of how to use the **Ranking Method** for identifying gaze-shift (saccade) and non-gaze-shift phases in mobile eye-tracking data.  
The method requires **(1) a gaze signal file** and **(2) an environment video** as input.


Download all files in this folder (the Ranking Method subdirectory of the repository) and place them next to your own script.

Import the Ranking method into your own script by the following lines:
```
from rankingMethod import runRankingMethod

from dataReader import readData as rankingReader
```

*Note*: If your script and the Ranking Method files are in different directories, you must import using the folder path: `from folder1.folder2.rankingMethod import ...`.

## Required Dataset Structure

Place your recordings inside a main dataset directory structured as follows:

```
DATASET/
│
├── recording_1/                # Name can be arbitrary
│   ├── gaze.txt                # Required filename
│   ├── world.mp4               # Required filename
│   └── world image times.txt   # Required filename
│
├── recording_2/
│   ├── gaze.txt
│   ├── world.mp4
│   └── world image times.txt
│
└── ...
```

**IMPORTANT**: We highly recommend using a blind detection and appending the labels in the last column. E.g., Pupil eye-trackers have easy-to-use blink detection that you can use. Having the blinks, can significanlty affect the performance of our method. In many mobile eye-trackers blinks are shows and fast jumps in the gaze, which are easy to be mistaken with saccades/gaze-shifts. `gaze.txt` file should have four columns: `[timestamp, gaze_x, gaze_y, blink(0/-1)]`. For blink column, -1 corresponds to blink. Please make sure that the blink column is included in the `gaze.txt` with this structure. Otherwise, if you'd like a fast, lower accuracy result, set all the blink column values to zero.


Then call the data reader:
```
gazeData, videosList, frameTimes = rankingReader(INP_DIR, "VU")
```

Replace the `INP_DIR` with your own dataset directory. The value `"VU"` refers to the data structure desribe above. By selecting `"VU"`, you let the data loader know that you have saved your files as instructed. `"VU"` adheres to the structure of exported recordings from Pupil eye-trackers. So you can use them directly with Ranking method.


If you would like to use your own data with different structure of data, you'll have to adapt the `rankingReader()` function. The outputs of `rankingReader()` function has the following structures. 



`gazeData`, includes the gaze information:
```
array([[ timestamp, gaze_x  , gaze_y  ,   blink(0/-1)],...]),
array([[ timestamp, gaze_x  , gaze_y  ,   blink(0/-1)],...]),
.
.
.
```

`videosList`, includes the address to the environment video for each recording:
```
'address/to/the/recording_1/world.mp4'
'address/to/the/recording_2/world.mp4'
.
.
.
```

`frameTimes`, includes timestamp of each frame for the environment video: 
```
[float,float, ....]
[float,float, ....]
[float,float, ....]
```

When you have all the mentioned variable read from the dataset, either by our datareader or yours, call the Ranking method to identify gaze-shift/saccades in your recording:

```
preds = runRankingMethod(gazeData, videosList, frameTimes, fast=False)
```

This primary function of Ranking method has an input `fast` that can be either set to `False` or `True`. Ranking method can take a long time because of the computations for the video frame contents. By setting `fast` to `True` your will significantly improve the compuatation time by slightly sacrificing the accuracy. This is done by using only the gaze probability for the gaze-shift detection without computing the visual scene probabilities. 

The identified gaze-shifts/saccades will be predicted as 1 in the `preds` variable. It is a two-dimentional matrix, each row are the predictions for each recording. Each index in each row, corresponds to a gaze point in the gaze signal.


### Citation
If you have used Ranking method in your work, we appreciate citing our manuscript:

```
citation will be provided
````

### Questions
If you have any question regarding how to execute Ranking method, open an Issue in GitHub or visit my website [[nejad.info](https://nejad.info/)] and use my contact information to get in touch with me 