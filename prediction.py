import cv2
import mediapipe as mp
import numpy as np
import joblib

# ========== LOAD MODEL ==========
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ========== MEDIAPIPE ==========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# ========== WEBCAM ==========
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    prediction = ""
    prev_prediction = ""
    counter = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmark_list = []

            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

            if len(landmark_list) == 63:

                # 🔥 SAME NORMALIZATION (MUST MATCH TRAINING)
                data = np.array(landmark_list).reshape(21, 3)

                base_x, base_y, base_z = data[0]

                normalized = []
                for x, y_val, z in data:
                    normalized.extend([x - base_x, y_val - base_y, z - base_z])

                normalized = np.array(normalized).reshape(1, -1)

                # Predict
                pred = model.predict(normalized)
                current_pred = label_encoder.inverse_transform(pred)[0]

                if current_pred == prev_prediction:
                    counter += 1
                else:
                    counter = 0

                prev_prediction = current_pred

                if counter > 5:  # stable detection
                    prediction = current_pred

    # Display prediction
    cv2.putText(img, f"Gesture: {prediction}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Gesture Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()