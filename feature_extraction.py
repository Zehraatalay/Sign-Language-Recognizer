import numpy as np


def extract_hand_features(hand_landmarks) -> np.ndarray:
    """
    Extract 42 position-normalized features from 21 hand landmarks.

    Feature order:
        [x0-x_min, y0-y_min, ..., x20-x_min, y20-y_min]

    This preserves the preprocessing used to train the original models.
    """

    landmarks = hand_landmarks.landmark

    if len(landmarks) != 21:
        raise ValueError(
            f"Expected 21 hand landmarks, received {len(landmarks)}."
        )

    x_coords = [landmark.x for landmark in landmarks]
    y_coords = [landmark.y for landmark in landmarks]

    min_x = min(x_coords)
    min_y = min(y_coords)

    features = []

    for landmark in landmarks:
        features.extend(
            [
                landmark.x - min_x,
                landmark.y - min_y,
            ]
        )

    return np.asarray(features, dtype=np.float64)