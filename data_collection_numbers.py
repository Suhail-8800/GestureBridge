# import cv2
# import mediapipe as mp
# import numpy as np
# import os

# # ========== CONFIG ==========
# current_gesture = "0"
# max_samples = 50
# sample_count = 0

# # Create base folder
# os.makedirs("data_numbers", exist_ok=True)

# # ========== MEDIAPIPE ==========
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(max_num_hands=2)
# mp_draw = mp.solutions.drawing_utils

# # ========== WEBCAM ==========
# cap = cv2.VideoCapture(0)

# print("Press keys 0-9 to change gesture")
# print("0–5 → Use ONE hand")
# print("6–9 → Use TWO hands")
# print("Press 's' to save data")
# print("Press 'q' to quit")

# while True:
#     success, img = cap.read()
#     if not success:
#         break

#     img = cv2.flip(img, 1)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = hands.process(img_rgb)

#     key = cv2.waitKey(1) & 0xFF

#     # 🔥 Change gesture (0–9)
#     if 48 <= key <= 57:
#         current_gesture = chr(key)
#         sample_count = 0
#         os.makedirs(f"data_numbers/{current_gesture}", exist_ok=True)
#         print(f"Switched to: {current_gesture}")

#     if results.multi_hand_landmarks:

#         all_landmarks = []

#         # Draw all detected hands
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#             for lm in hand_landmarks.landmark:
#                 all_landmarks.extend([lm.x, lm.y, lm.z])

#         # 🔥 CASE 1: ONE HAND (0–5)
#         if len(results.multi_hand_landmarks) == 1 and current_gesture in ['0','1','2','3','4','5']:

#             if len(all_landmarks) == 63:
#                 # Duplicate to make 126
#                 combined = all_landmarks + all_landmarks

#                 if key == ord('s') and sample_count < max_samples:
#                     np.save(f"data_numbers/{current_gesture}/{sample_count}.npy", combined)
#                     sample_count += 1
#                     print(f"{current_gesture}: {sample_count}/{max_samples}")

#         # 🔥 CASE 2: TWO HANDS (6–9)
#         elif len(results.multi_hand_landmarks) == 2 and current_gesture in ['6','7','8','9']:

#             if len(all_landmarks) == 126:
#                 if key == ord('s') and sample_count < max_samples:
#                     np.save(f"data_numbers/{current_gesture}/{sample_count}.npy", all_landmarks)
#                     sample_count += 1
#                     print(f"{current_gesture}: {sample_count}/{max_samples}")

#     # Display info
#     cv2.putText(img, f"Number: {current_gesture}", (10, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     cv2.putText(img, f"Samples: {sample_count}/{max_samples}", (10, 80),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#     cv2.imshow("Number Data Collection (0-9)", img)

#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



import cv2
import mediapipe as mp
import numpy as np
import os

# ========== CONFIG ==========
current_gesture = "0"
max_samples = 50
sample_count = 0

os.makedirs("data_numbers", exist_ok=True)

# ========== MEDIAPIPE ==========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# ========== WEBCAM ==========
cap = cv2.VideoCapture(0)

print("Press keys 0-9 to change gesture")
print("Press 'x' for 10")
print("0–5 → ONE hand")
print("6–10 → TWO hands")
print("Press 's' to save")
print("Press 'q' to quit")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    key = cv2.waitKey(1) & 0xFF

    # 🔥 Change gesture (0–9)
    if 48 <= key <= 57:
        current_gesture = chr(key)
        sample_count = 0
        os.makedirs(f"data_numbers/{current_gesture}", exist_ok=True)
        print(f"Switched to: {current_gesture}")

    # 🔥 Add 10 (press X)
    elif key == ord('x'):
        current_gesture = "10"
        sample_count = 0
        os.makedirs(f"data_numbers/{current_gesture}", exist_ok=True)
        print("Switched to: 10")

    if results.multi_hand_landmarks:

        all_landmarks = []

        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for lm in hand_landmarks.landmark:
                all_landmarks.extend([lm.x, lm.y, lm.z])

        # ==========================
        # 🔢 CASE 1: ONE HAND (0–5)
        # ==========================
        if len(results.multi_hand_landmarks) == 1 and current_gesture in ['0','1','2','3','4','5']:

            if len(all_landmarks) == 63:
                combined = all_landmarks + all_landmarks  # duplicate

                if key == ord('s') and sample_count < max_samples:
                    np.save(f"data_numbers/{current_gesture}/{sample_count}.npy", combined)
                    sample_count += 1
                    print(f"{current_gesture}: {sample_count}/{max_samples}")

        # ==========================
        # 🔢 CASE 2: TWO HANDS (6–10)
        # ==========================
        elif len(results.multi_hand_landmarks) == 2 and current_gesture in ['6','7','8','9','10']:

            if len(all_landmarks) == 126:
                if key == ord('s') and sample_count < max_samples:
                    np.save(f"data_numbers/{current_gesture}/{sample_count}.npy", all_landmarks)
                    sample_count += 1
                    print(f"{current_gesture}: {sample_count}/{max_samples}")

    # Display
    cv2.putText(img, f"Number: {current_gesture}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(img, f"Samples: {sample_count}/{max_samples}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Number Data Collection (0-10)", img)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()