import cv2
import mediapipe as mp
import numpy as np
import os

# ========== CONFIG ==========
max_samples = 50
sample_count = 0

# Word mapping (KEY → WORD)
word_map = {
    'h': 'HELLO',
    'y': 'YES',
    'n': 'NO',
    'z': 'STOP',
    'g': 'GO',
    'w': 'WAIT',
    'e': 'HELP',
    'l': 'LOVE',
    'o': 'OK',
    'a': 'NAME',
    'p': 'PAYMENT',
    'r': 'SORRY',
    'x': 'PLEASE'
}

# Words requiring TWO HANDS
two_hand_words = ['HELP']

current_word = "HELLO"

os.makedirs("data_words", exist_ok=True)

# ========== MEDIAPIPE ==========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# ========== WEBCAM ==========
cap = cv2.VideoCapture(0)

print("===== WORD DATA COLLECTION =====")
print("Press keys to switch words:")
for k, v in word_map.items():
    print(f"{k.upper()} → {v}")
print("Press 'S' to save sample")
print("Press 'Q' to quit")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    key = cv2.waitKey(1) & 0xFF

    # 🔥 Switch word
    if chr(key).lower() in word_map:
        current_word = word_map[chr(key).lower()]
        sample_count = 0
        os.makedirs(f"data_words/{current_word}", exist_ok=True)
        print(f"Switched to: {current_word}")

    if results.multi_hand_landmarks:

        all_landmarks = []

        # 🔥 SORT HANDS (CRITICAL)
        hands_sorted = sorted(
            results.multi_hand_landmarks,
            key=lambda h: h.landmark[0].x
        )

        # Draw + extract
        for hand_landmarks in hands_sorted:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for lm in hand_landmarks.landmark:
                all_landmarks.extend([lm.x, lm.y, lm.z])

        combined = None

        # ==========================
        # 🔥 ONE HAND WORDS
        # ==========================
        if current_word not in two_hand_words and len(all_landmarks) == 63:
            combined = all_landmarks + all_landmarks  # duplicate → 126

        # ==========================
        # 🔥 TWO HAND WORDS
        # ==========================
        elif current_word in two_hand_words and len(all_landmarks) == 126:
            combined = all_landmarks

        # ==========================
        # 🔥 SAVE
        # ==========================
        if combined is not None:
            if key == ord('s') and sample_count < max_samples:
                np.save(f"data_words/{current_word}/{sample_count}.npy", combined)
                sample_count += 1
                print(f"{current_word}: {sample_count}/{max_samples}")

    # ==========================
    # DISPLAY
    # ==========================
    cv2.putText(img, f"WORD: {current_word}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(img, f"Samples: {sample_count}/{max_samples}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Word Data Collection", img)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()