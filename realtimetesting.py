from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import mediapipe as mp

from feature_extraction import extract_hand_features


MODEL_FILES = {
    "random_forest": ("Random Forest", "model.p"),
    "knn": ("KNN", "model_knn.p"),
    "svm": ("SVM", "svm_model.p"),
    "logistic_regression": ("Logistic Regression", "logistic_model.p"),
    "decision_tree": ("Decision Tree", "decision_tree_model.p"),
    "naive_bayes": ("Naive Bayes", "naive_bayes_model.p"),
}


def load_models(models_dir: Path, selected_model: str) -> dict:
    """Load the requested pretrained classifiers."""

    if selected_model == "all":
        selected_names = list(MODEL_FILES)
    else:
        selected_names = [selected_model]

    models = {}

    for model_name in selected_names:
        display_name, filename = MODEL_FILES[model_name]
        model_path = models_dir / filename

        if not model_path.is_file():
            if selected_model == "all":
                print(f"[SKIP] {display_name}: model file not found.")
                continue

            raise FileNotFoundError(f"Model file not found: {model_path}")

        with model_path.open("rb") as file:
            saved = pickle.load(file)

        model = saved["model"] if isinstance(saved, dict) else saved

        models[display_name] = model

    if not models:
        raise ValueError(f"No models found in: {models_dir}")

    return models


def predict_sign(features, models: dict) -> dict[str, str]:
    """Predict a sign using the selected classifiers."""

    # Scikit-learn expects a 2D input: (samples, features).
    X = features.reshape(1, -1)

    predictions = {}

    for model_name, model in models.items():
        prediction = model.predict(X)[0]

        # Models were trained using string labels such as "A", "B", "C".
        predictions[model_name] = str(prediction)

    return predictions


def run_webcam(models: dict, camera_index: int = 0) -> None:
    """Run real-time hand sign recognition using the webcam."""

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Cannot open camera {camera_index}. "
            "Check camera permissions and device availability."
        )

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            failed_frames = 0
            max_failed_frames = 10
            while True:
                success, frame = cap.read()

                if not success:
                    failed_frames += 1

                    if failed_frames >= max_failed_frames:
                        print("Camera stopped providing frames.")
                        break

                    continue

                failed_frames = 0
                # Mirror the webcam image for a more natural preview.
                frame = cv2.flip(frame, 1)

                height, width = frame.shape[:2]

                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    features = extract_hand_features(hand_landmarks)
                    predictions = predict_sign(features, models)

                    x_coords = [
                        landmark.x for landmark in hand_landmarks.landmark
                    ]
                    y_coords = [
                        landmark.y for landmark in hand_landmarks.landmark
                    ]

                    x1 = max(0, int(min(x_coords) * width) - 10)
                    y1 = max(0, int(min(y_coords) * height) - 10)
                    x2 = min(width - 1, int(max(x_coords) * width) + 10)
                    y2 = min(height - 1, int(max(y_coords) * height) + 10)

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 0),
                        2,
                    )

                    # Show predictions in a fixed position to prevent
                    # text from moving outside the frame.
                    for index, (model_name, prediction) in enumerate(
                        predictions.items()
                    ):
                        text = f"{model_name}: {prediction}"

                        cv2.putText(
                            frame,
                            text,
                            (15, 35 + index * 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                cv2.imshow("Sign Language Recognition", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time ASL recognition using pretrained ML models."
    )

    parser.add_argument(
        "--models-dir",
        type=Path,
        required=True,
        help="Directory containing pretrained model files.",
    )

    parser.add_argument(
        "--model",
        choices=[*MODEL_FILES.keys(), "all"],
        default="random_forest",
        help="Classifier to use. Defaults to random_forest.",
    )

    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera device index. Defaults to 0.",
    )

    args = parser.parse_args()

    models = load_models(
        models_dir=args.models_dir,
        selected_model=args.model,
    )

    print("Loaded models:", ", ".join(models))
    print("Press 'q' to quit.")

    run_webcam(
        models=models,
        camera_index=args.camera_index,
    )


if __name__ == "__main__":
    main()