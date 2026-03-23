import cv2
import mediapipe as mp
import numpy as np
import os

# ========== CONFIG ==========
current_gesture = "A"
max_samples = 50
sample_count = 0

# Create base data folder
os.makedirs("data", exist_ok=True)

# ========== MEDIAPIPE ==========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# ========== WEBCAM ==========
cap = cv2.VideoCapture(0)

print("Press keys A-Z to change gesture")
print("Press 's' to save data")
print("Press 'q' to quit")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    key = cv2.waitKey(1) & 0xFF

    # 🔥 Change gesture (A-Z)
    if 65 <= key <= 90:  # ASCII A-Z
        current_gesture = chr(key)
        sample_count = 0
        os.makedirs(f"data/{current_gesture}", exist_ok=True)
        print(f"Switched to gesture: {current_gesture}")

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

            if len(landmark_list) == 63:

                # Save data
                if key == ord('s') and sample_count < max_samples:
                    np.save(f"data/{current_gesture}/{sample_count}.npy", landmark_list)
                    sample_count += 1
                    print(f"{current_gesture}: {sample_count}/{max_samples}")

    # Display info
    cv2.putText(img, f"Gesture: {current_gesture}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(img, f"Samples: {sample_count}/{max_samples}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Data Collection", img)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()