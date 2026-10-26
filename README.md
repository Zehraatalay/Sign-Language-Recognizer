# Sign Language Recognizer

Real-time recognition of **24 static American Sign Language (ASL) letters** using MediaPipe and traditional machine learning.

The system extracts hand landmarks, classifies gestures using six ML algorithms, and displays predictions through a webcam.

## Approach

**Hand Detection → Feature Extraction → Classification → Real-Time Prediction**

- **MediaPipe Hands:** Detects 21 hand landmarks.
- **Feature Engineering:** Converts landmark coordinates into 42 translation-normalized features.
- **Machine Learning:** Compares Random Forest, KNN, SVM, Logistic Regression, Decision Tree, and Gaussian Naive Bayes.
- **OpenCV:** Enables real-time webcam inference.

## Model Performance

Models were trained and evaluated separately on two datasets.

| Model | MyDataset | Roboflow |
|:---|---:|---:|
| **Random Forest** | **99.32%** | **92.18%** |
| KNN | 99.32% | 91.50% |
| SVM | 97.97% | 91.84% |
| Decision Tree | 98.31% | 79.59% |
| Logistic Regression | 92.06% | 77.89% |
| Naive Bayes | 86.82% | 65.99% |

*Test accuracy (%). Random Forest achieved the highest accuracy on both datasets.*

Full evaluation metrics: [MyDataset](results/mydataset_metrics.csv) · [Roboflow](results/roboflow_metrics.csv)

## Getting Started

**Requirements:** Python 3.11

```bash
pip install -r requirements.txt
```

### 1. Extract Features

```bash
python dataprocessing.py --dataset-dir /path/to/dataset --output-dir /path/to/processed
```

### 2. Train Models

```bash
python train_models.py --dataset-dir /path/to/processed --model all --output-dir trained_models
```

### 3. Evaluate Models

```bash
python evaluate_models.py --dataset-dir /path/to/processed --models-dir trained_models --split test
```

### 4. Run Webcam Recognition

```bash
python realtimetesting.py --models-dir trained_models --model random_forest
```

Use `--model all` to compare all six classifiers.

## Limitations

- Supports 24 static ASL letters; motion-based **J** and **Z** are excluded.
- Processes one hand at a time.
- Dataset test accuracy may not reflect performance in real-world webcam conditions.

---

**Tech Stack:** Python · MediaPipe · OpenCV · NumPy · scikit-learn
