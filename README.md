# lane-detection
Simple lane detection for road images

# Environment setup

```bash
$ wget https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh
$ bash Anaconda3-4.2.0-Linux-x86_64.sh
$ source activate root
(root) $ conda install -c https://conda.anaconda.org/menpo opencv3
(root) $ pip install moviepy
(root) $ pip install click
(root) $ pip install imutils
```

You should be able to import cv2 and moviepy

# Usage
```bash
python detect_lanes.py -i --input_file_name input_images/solidWhiteRight.jpg --output_file_name output_images/output_solidWhiteRight.jpg
```
![alt tag](https://raw.githubusercontent.com/hristo-vrigazov/lane-detection/master/input_images/solidWhiteCurve.jpg)
![alt tag](https://raw.githubusercontent.com/hristo-vrigazov/lane-detection/master/output_images/output_solidWhiteRight.jpg)
