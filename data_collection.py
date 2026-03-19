import cv2
import mediapipe as mp
import numpy as np
import os

# ========== CONFIG ==========
gesture_name = "A"
save_path = f"data/{gesture_name}"
os.makedirs(save_path, exist_ok=True)

max_samples = 50  # number of samples to collect
sample_count = 0

# ========== MEDIAPIPE ==========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# ========== WEBCAM ==========
cap = cv2.VideoCapture(0)

print("Press 's' to save data")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

            if len(landmark_list) == 63:

                key = cv2.waitKey(1)

                if key == ord('s') and sample_count < max_samples:
                    np.save(f"{save_path}/{sample_count}.npy", landmark_list)
                    sample_count += 1
                    print(f"Saved sample {sample_count}")

    cv2.putText(img, f"Samples: {sample_count}/{max_samples}", 
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)

    cv2.imshow("Data Collection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()