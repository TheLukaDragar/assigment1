# Dataset split
We split the dataset into train, and test 

test set contains identities 008, 004, 043, 022, 035, 034, 014, 028, 042, 025
and composes of 100 images

while the train set contains the rest of the identities and composes of 400 images

# VJ detection

on the test set we optimized VJ detection parameters to maximize iou
when multiple ears were detected we only chose the biggest one

the grid search was done on the following parameters:
```python
grid = {
    "scaleFactor": list(i for i in np.arange(1.01, 1.1, 0.01)),
    "minNeighbors": [2, 4, 6],
    "minSize": [20, 30, 40],
}
```
the best parameters were:

```python
Highest avg iou at index 11
Avg iou: 0.3451
Scale factor: 1.02
minNeighbors: 2
minSize: 30
Detected: 323
Not detected: 177
Avg iou detected: 0.4274
```

[image1]: ./analysis_VJ_highest_avg_iou.png "VJ highest avg iou"
![VJ highest avg iou][image1]

# LBP feature extraction
then we proceded to implement varous LBP impelementations and did a grid search on the following parameters:
```python
grid = {
    "lbptype": ["default", "uniform", "rotation"],
    "radius": [1, 2],
    "neighbors": [8, 16],
    "histograms": [False, True],
    "resize_to": [24,64, 128],
    "use_lib": [False, True],
    "use_raw": [False, True]
}
```

the best parameters were:

```python
Highest score at index 14
Score: 0.2
Best parameter set: {'histograms': False, 'lbptype': 'default', 'neighbors': 8, 'radius': 2, 'resize_to': 24, 'use_lib': True, 'use_raw': False, 'score': 0.2}
```

Our implementation is also in the top 5:

```python
Top 5:
14: 0.2 - histograms: False, lbptype: default, neighbors: 8, radius: 2, resize_to: 24, use_lib: True, use_raw: False
158: 0.2 - histograms: True, lbptype: default, neighbors: 8, radius: 2, resize_to: 24, use_lib: True, use_raw: False
108: 0.1975 - histograms: False, lbptype: rotation, neighbors: 8, radius: 2, resize_to: 24, use_lib: False, use_raw: False
146: 0.1875 - histograms: True, lbptype: default, neighbors: 8, radius: 1, resize_to: 24, use_lib: True, use_raw: False
2: 0.1875 - histograms: False, lbptype: default, neighbors: 8, radius: 1, resize_to: 24, use_lib: True, use_raw: False
```


[image2]: ./LBP_grid_search.png "LBP highest score"
![LBP highest score][image2]




We chose to use our implementation with the following parameters for the rest of the experiments:
```python
histograms: False, lbptype: rotation, neighbors: 8, radius: 2, resize_to: 24, use_lib: False, use_raw: False
```

# Test set results VJ_LBP_test.py
## VJ 
```python
{'detected': 72, 'not_detected': 28, 'avg_iou': 'tensor(0.2818, dtype=torch.float64)', 'avg_iou_detected': 'tensor(0.3915, dtype=torch.float64)'}
```


VJ achieved 72% detection rate with an average iou of 0.2818 and an average iou of 0.3915 on the detected ears.

## LBP with out lbp implementation

```python
Extracting LBP features for 72 images
Extracting LBP features for 100 images
Features array shape: (72, 400)
GT Features array shape: (100, 400)

Accuracy using ground truth VJ detections: 0.36 36/100

number of detected ears: 72/100

Accuracy using best VJ detections when considering only detected ears: 0.1527777777777778 11/72
Accuracy using best VJ detections when considering all images: 0.11 11/100
```



## comparison with using only raw image vectors
```python

Extracting LBP features for 100 images
Extracting LBP features for 72 images
Raw Features array shape: 72
Raw GT Features array shape: 100
Accuracy with ground truth raw ear image comparison: 0.27 27/100
Accuracy with best VJ, raw ear image comparison when considering only detected ears: 0.18055555555555555 13/72
Accuracy with best VJ, raw ear image comparison when considering all images: 0.13 13/100

```








