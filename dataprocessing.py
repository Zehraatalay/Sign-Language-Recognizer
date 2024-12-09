import os
import cv2
import pickle
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands

# Ana klasör yolu
DATA_DIR = "/Users/zehraatalay/Desktop/dataset"

# İşlenecek alt klasörler
SUBSETS = ["train", "validation", "test"]

# Mediapipe modeli
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

for subset in SUBSETS:
    subset_dir = os.path.join(DATA_DIR, subset)

    data = []
    labels = []

    print(f"Processing {subset} data...")

    for dir_ in os.listdir(subset_dir):
        dir_path = os.path.join(subset_dir, dir_)

        # Sadece alt klasörleri işleme
        if os.path.isdir(dir_path):
            for img_name in os.listdir(dir_path):
                img_path = os.path.join(dir_path, img_name)

                # Görüntüyü okuyun ve işleyin
                try:
                    img = cv2.imread(img_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    result = hands.process(img_rgb)

                    if result.multi_hand_landmarks:
                        data_aux = []
                        x_ = []
                        y_ = []

                        for hand_landmarks in result.multi_hand_landmarks:
                            for i in range(len(hand_landmarks.landmark)):
                                x = hand_landmarks.landmark[i].x
                                y = hand_landmarks.landmark[i].y
                                x_.append(x)
                                y_.append(y)

                            for i in range(len(hand_landmarks.landmark)):
                                x = hand_landmarks.landmark[i].x
                                y = hand_landmarks.landmark[i].y

                                data_aux.append(x - min(x_))
                                data_aux.append(y - min(y_))

                        data.append(data_aux)
                        labels.append(dir_)  # Sınıf adı klasör adından alınır

                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")

    # Pickle dosyasını kaydet
    pickle_path = os.path.join(DATA_DIR, f"{subset}_data.pickle")
    with open(pickle_path, "wb") as f:
        pickle.dump({"data": data, "labels": labels}, f)

    print(f"{subset.capitalize()} data saved to {pickle_path}")
