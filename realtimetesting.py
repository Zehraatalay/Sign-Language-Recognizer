import pickle
import cv2
import mediapipe as mp
import numpy as np

# Modellerin yollarını tanımlayın
DATA_DIR = "/Users/zehraatalay/Desktop/dataset"
model_paths = {
    "Random Forest": f"{DATA_DIR}/model.p",
    "Decision Tree": f"{DATA_DIR}/decision_tree_model.p",
    "Logistic Regression": f"{DATA_DIR}/logistic_model.p",
    "SVM": f"{DATA_DIR}/svm_model.p",
    "KNN": f"{DATA_DIR}/model_knn.p",
    "Naive Bayes": f"{DATA_DIR}/naive_bayes_model.p",
}

labels_dict = {
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6,
    "H": 7, "I": 8, "K": 9, "L": 10, "M": 11, "N": 12, "O": 13,
    "P": 14, "Q": 15, "R": 16, "S": 17, "T": 18, "U": 19,
    "V": 20, "W": 21, "X": 22, "Y": 23
}


# Tüm modelleri yükleyin
models = {}
for model_name, path in model_paths.items():
    with open(path, "rb") as f:
        models[model_name] = pickle.load(f)["model"]


# Mediapipe kurulumları
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Video yakalama
cap = cv2.VideoCapture(0)

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Mediapipe ile el işleme
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            # Landmark'ları normalize et
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

            # Çerçeve koordinatları
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            # Her modelin tahmini
            predictions = {}
            for model_name, model in models.items():
                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    predictions[model_name] = labels_dict[int(prediction[0])]
                except Exception as e:
                    predictions[model_name] = f"Error: {str(e)}"

            # Çerçeve üzerine yazdırma
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            y_offset = y1 - 10
            font_scale = 1.0  # Daha büyük font için artırıldı
            font_thickness = 2  # Daha kalın font
            for model_name, prediction in predictions.items():
                cv2.putText(
                    frame,
                    f"{model_name}: {prediction}",
                    (x1, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,  # Font büyüklüğü
                    (255, 255, 255),
                    font_thickness,  # Font kalınlığı
                    cv2.LINE_AA,
                )
                y_offset -= int(30 * font_scale)  # Satırlar arası mesafe artırıldı

    # Çerçeveyi göster
    cv2.imshow("Real-time Model Predictions", frame)

    # Çıkış için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Kaynakları serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
