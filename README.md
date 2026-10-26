Sign Language Recognizer

A machine learning-based system for recognizing 24 static American Sign Language (ASL) letters using MediaPipe hand landmarks and scikit-learn classifiers.

Supports real-time recognition through a webcam.

How It Works

1. Hand Detection: MediaPipe detects 21 hand landmarks.
2. Feature Extraction: Landmark coordinates are transformed into 42 translation-normalized features.
3. Classification: Six machine learning models are trained and evaluated.
4. Real-Time Inference: OpenCV displays predictions from the selected model.

Models: Random Forest · KNN · SVM · Logistic Regression · Decision Tree · Gaussian Naive Bayes

Results

Models were trained and evaluated separately on two datasets.

Model	MyDataset Accuracy	Roboflow Accuracy
Random Forest	99.32%	92.18%
KNN	99.32%	91.50%
SVM	97.97%	91.84%
Decision Tree	98.31%	79.59%
Logistic Regression	92.06%	77.89%
Naive Bayes	86.82%	65.99%

Best-performing model: Random Forest on both datasets.

Full evaluation metrics (accuracy, precision, recall, and macro F1) are available in results/.

Installation

Requirements: Python 3.11

pip install -r requirements.txt

Usage

1. Extract features

python dataprocessing.py \
  --dataset-dir /path/to/dataset \
  --output-dir /path/to/processed

2. Train models

python train_models.py \
  --dataset-dir /path/to/processed \
  --model all \
  --output-dir trained_models

3. Evaluate models

python evaluate_models.py \
  --dataset-dir /path/to/processed \
  --models-dir trained_models \
  --split test

4. Run real-time recognition

python realtimetesting.py \
  --models-dir trained_models \
  --model random_forest

Use --model all to compare predictions from all six classifiers.

Limitations

* Recognizes 24 static ASL letters; motion-based J and Z are excluded.
* Supports single-hand recognition.
* Test accuracy may not reflect performance under different real-world conditions.

⸻

Tech Stack: Python · MediaPipe · OpenCV · NumPy · scikit-learn