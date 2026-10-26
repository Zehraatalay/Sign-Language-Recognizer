Sign Language Recognizer

A machine learning project for recognizing 24 static American Sign Language (ASL) letters using MediaPipe hand landmarks and scikit-learn classifiers. The system supports real-time recognition through a webcam.

How It Works

1. Hand Detection: MediaPipe extracts 21 hand landmarks.
2. Feature Extraction: Landmark coordinates are converted into 42 position-normalized features.
3. Classification: Six ML models are trained and evaluated.
4. Real-Time Inference: OpenCV displays predictions from the selected model.

Models: Random Forest, KNN, SVM, Logistic Regression, Decision Tree, and Gaussian Naive Bayes.

Results

Models were trained and evaluated separately on two datasets.

Model	MyDataset Accuracy	Roboflow Accuracy
Random Forest	99.32%	92.18%
KNN	99.32%	91.50%
SVM	97.97%	91.84%
Decision Tree	98.31%	79.59%
Logistic Regression	92.06%	77.89%
Naive Bayes	86.82%	65.99%

Random Forest achieved the highest accuracy on both datasets.

Detailed evaluation results are available in the results/ directory.

Installation

Requires Python 3.11.

pip install -r requirements.txt

Usage

Prepare features:

python dataprocessing.py --dataset-dir /path/to/dataset --output-dir /path/to/processed

Train models:

python train_models.py --dataset-dir /path/to/processed --model all --output-dir trained_models

Evaluate models:

python evaluate_models.py --dataset-dir /path/to/processed --models-dir trained_models --split test

Run webcam recognition:

python realtimetesting.py --models-dir trained_models --model random_forest

Use --model all to compare all six classifiers.

Limitations

* Recognizes 24 static ASL letters; motion-based J and Z are excluded.
* Processes one hand at a time.
* Test accuracy does not necessarily reflect performance in real-world webcam conditions.

Tech Stack: Python · MediaPipe · OpenCV · NumPy · scikit-learn