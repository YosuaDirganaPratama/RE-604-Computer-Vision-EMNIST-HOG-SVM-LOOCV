# EMNIST Letter Classification using HOG and SVM

## Project Description

This project implements handwritten character classification using the EMNIST (Extended MNIST) Letters dataset with:

- HOG (Histogram of Oriented Gradients) as feature extraction
- SVM (Support Vector Machine) as classifier
- Grid Search for parameter tuning
- Leave One Out Cross Validation (LOOCV) for evaluation

This project was developed for Computer Vision examination/project purposes.

---

## Dataset

Dataset source:

https://www.kaggle.com/datasets/crawford/emnist/data

Dataset used:

- emnist-letters-train.csv
- emnist-letters-test.csv
- emnist-letters-mapping.txt

Dataset location:

```text
D:/Computer Vision/EMNIST/datasets/
```

Project structure:

```text
EMNIST/
│
├── datasets/
│   ├── emnist_source_files/
│   ├── results/
│   ├── emnist-letters-train.csv
│   ├── emnist-letters-test.csv
│   └── emnist-letters-mapping.txt
│
├── emnist_hog_svm_loocv.ipynb
├── README.md
└── requirements.txt
```

---

## Installation

Install required libraries:

```bash
pip install matplotlib
pip install numpy
pip install seaborn
pip install scikit-learn
pip install scikit-image
pip install pandas
pip install mlxtend
```

or:

```bash
pip install -r requirements.txt
```

---

## Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

from skimage.feature import hog

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
```

---

## Dataset Preparation

Dataset preprocessing steps:

1. Read EMNIST CSV dataset
2. Shuffle dataset
3. Convert labels into range 0–25
4. Select balanced dataset:
   - 26 classes
   - 100 samples each class
5. Total samples:

```text
2600 samples
```

Split data:

```text
Training : 80%
Testing : 20%
```

Result:

```text
X_train : 2080
X_test : 520
```

---

## HOG Feature Extraction

Modified HOG parameters:

```python
orientations=8

pixels_per_cell=(4,4)

cells_per_block=(2,2)

block_norm='L2'
```

Extract HOG features:

```python
X_train_hog=hog_features(X_train)

X_test_hog=hog_features(X_test)
```

---

## Support Vector Machine

SVM parameters:

```python
param_grid={

'C':[0.1,1,10],

'kernel':['linear','rbf'],

'gamma':['scale','auto']

}
```

Grid Search:

```python
grid_search=GridSearchCV(
svm,
param_grid,
cv=5,
n_jobs=-1,
verbose=2
)
```

Best parameter result:

```text
Best Parameter:

{'C':10,
'gamma':'scale',
'kernel':'rbf'}
```

---

## Leave One Out Cross Validation (LOOCV)

LOOCV was used for performance evaluation.

```python
loo=LeaveOneOut()
```

Prediction performed on testing dataset.

---

## Evaluation Metrics

Metrics used:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Example:

```text
Accuracy : 0.84
Precision : 0.84
Recall : 0.84
F1-score : 0.84
```

---

## Results

Confusion Matrix:

(Add screenshot here)

Prediction Results:

(Add screenshot here)

---

## Output Example

True Label and Prediction:

```text
True : a
Prediction : a

True : c
Prediction : c

True : g
Prediction : q
```

---

## Author

Name: Yosua Dirgana Pratama
Nim : 4222301009
Course: Computer Vision

Project: EMNIST Letter Classification using HOG and SVM