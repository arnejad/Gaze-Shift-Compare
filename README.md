# Detecting Gaze Shifts of Moving Observers in Dynamic Environments

Welcome to the official repository for our paper:
“Detecting gaze shifts of moving observers in dynamic environments”
submitted to Behavior Research Methods.

This repository provides: 1) The Python code to reproduce our benchmark comparing six popular gaze-shift detection methods on head-mounted eye-tracking data recorded outdoors, 2) The implementation of our proposed Ranking algorithm for robust, parameter-free gaze-shift detection, 3) Scripts for evaluation, comparison, and visualization.



The public dataset can be found in the following link:

[https://unishare.nl/index.php/s/Ypgm3btwGs5wAYr](https://unishare.nl/index.php/s/Ypgm3btwGs5wAYr)

Before being able to run the code, write an script that creates a folder named `image_2` in each particiapnt folder and transforms each `world.mp4` video to a set of images from the frames of the video. The naming of the image frames should start from `000001.png`. 


The required libriaries and independencies are listed in the `requirements.txt`. You can use pip to install all of them together.

Set the address to the data folder in the `config.py`. 

For comparing all the methods including the pre-trained machine-learning-based methods, run `main.py`. For optimizing the threshold-based methodsm run `optimizeThreshold.py`. To retrain the machine-learning-based methods, execute `training.py`.

For citations please use the following [publication]():

```
Citation will be provided after publication
```

## Remark and Acknowledgement

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Sklodowska-Curie grant agreement No 955590.

<img src="./figs_stat/visio.png" height="60"> <img src="./figs_stat/VuA.png" height="60"> <img src="./figs_stat/eu_flag.jpg" height="60"> <img src="./figs_stat/rug.png" height="60"> <img src="./figs_stat/optivist.png" height="60">

