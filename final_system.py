# import cv2
# import mediapipe as mp
# import numpy as np
# import joblib

# # ========== LOAD MODELS ==========
# alpha_model = joblib.load("model.pkl")
# alpha_encoder = joblib.load("label_encoder.pkl")

# num_model = joblib.load("model_numbers_full.pkl")
# num_encoder = joblib.load("label_encoder_numbers_full.pkl")

# # ========== MEDIAPIPE ==========
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(max_num_hands=2)  # 🔥 allow 2 hands
# mp_draw = mp.solutions.drawing_utils

# # ========== MODE ==========
# mode = "ALPHABET"

# # Stability
# prev_prediction = ""
# counter = 0
# prediction = ""

# # ========== WEBCAM ==========
# cap = cv2.VideoCapture(0)

# print("Press 'M' to switch mode")
# print("Press 'Q' to quit")

# while True:
#     success, img = cap.read()
#     if not success:
#         break

#     img = cv2.flip(img, 1)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     results = hands.process(img_rgb)

#     key = cv2.waitKey(1) & 0xFF

#     # 🔥 MODE SWITCH
#     if key == ord('m'):
#         mode = "NUMBER" if mode == "ALPHABET" else "ALPHABET"
#         print(f"Switched to {mode} mode")

#     if results.multi_hand_landmarks:

#         all_landmarks = []

#         # Draw all hands and collect landmarks
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#             for lm in hand_landmarks.landmark:
#                 all_landmarks.extend([lm.x, lm.y, lm.z])

#         # ==========================
#         # 🔤 ALPHABET MODE (1 HAND)
#         # ==========================
#         if mode == "ALPHABET" and len(all_landmarks) >= 63:

#             data = np.array(all_landmarks[:63]).reshape(21, 3)

#             base_x, base_y, base_z = data[0]

#             normalized = []
#             for x, y_val, z in data:
#                 normalized.extend([x - base_x, y_val - base_y, z - base_z])

#             normalized = np.array(normalized).reshape(1, -1)

#             pred = alpha_model.predict(normalized)
#             current_pred = alpha_encoder.inverse_transform(pred)[0]

#         # ==========================
#         # 🔢 NUMBER MODE (0–9)
#         # ==========================
#         elif mode == "NUMBER":

#             # CASE 1: ONE HAND → duplicate
#             if len(all_landmarks) == 63:
#                 combined = all_landmarks + all_landmarks

#             # CASE 2: TWO HANDS → use both
#             elif len(all_landmarks) == 126:
#                 combined = all_landmarks

#             else:
#                 combined = None

#             if combined:

#                 data = np.array(combined).reshape(42, 3)

#                 base_x, base_y, base_z = data[0]

#                 normalized = []
#                 for x, y_val, z in data:
#                     normalized.extend([x - base_x, y_val - base_y, z - base_z])

#                 normalized = np.array(normalized).reshape(1, -1)

#                 pred = num_model.predict(normalized)
#                 current_pred = num_encoder.inverse_transform(pred)[0]
#             else:
#                 current_pred = ""

#         else:
#             current_pred = ""

#         # 🔥 STABILITY LOGIC
#         if current_pred == prev_prediction:
#             counter += 1
#         else:
#             counter = 0

#         prev_prediction = current_pred

#         if counter > 5:
#             prediction = current_pred

#     # DISPLAY
#     cv2.putText(img, f"Mode: {mode}", (10, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

#     cv2.putText(img, f"Output: {prediction}", (10, 90),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

#     cv2.imshow("GestureBridge System", img)

#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()






import cv2
import mediapipe as mp
import numpy as np
import joblib

# ========== LOAD MODELS ==========
alpha_model = joblib.load("model.pkl")
alpha_encoder = joblib.load("label_encoder.pkl")

num_model = joblib.load("model_numbers_full.pkl")
num_encoder = joblib.load("label_encoder_numbers_full.pkl")

# ========== MEDIAPIPE ==========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# ========== MODE ==========
mode = "ALPHABET"

# Stability
prev_prediction = ""
counter = 0
prediction = ""

# ========== WEBCAM ==========
cap = cv2.VideoCapture(0)

print("Press 'M' to switch mode")
print("Press 'Q' to quit")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    key = cv2.waitKey(1) & 0xFF

    # 🔥 MODE SWITCH
    if key == ord('m'):
        mode = "NUMBER" if mode == "ALPHABET" else "ALPHABET"
        print(f"Switched to {mode} mode")

    if results.multi_hand_landmarks:

        all_landmarks = []

        # 🔥 SORT HANDS (IMPORTANT for consistency)
        hands_sorted = sorted(
            results.multi_hand_landmarks,
            key=lambda h: h.landmark[0].x  # sort by wrist x
        )

        # Draw + extract
        for hand_landmarks in hands_sorted:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for lm in hand_landmarks.landmark:
                all_landmarks.extend([lm.x, lm.y, lm.z])

        # ==========================
        # 🔤 ALPHABET MODE
        # ==========================
        if mode == "ALPHABET" and len(all_landmarks) >= 63:

            data = np.array(all_landmarks[:63]).reshape(21, 3)

            base_x, base_y, base_z = data[0]

            normalized = []
            for x, y_val, z in data:
                normalized.extend([x - base_x, y_val - base_y, z - base_z])

            normalized = np.array(normalized).reshape(1, -1)

            pred = alpha_model.predict(normalized)
            current_pred = alpha_encoder.inverse_transform(pred)[0]

        # ==========================
        # 🔢 NUMBER MODE (0–10)
        # ==========================
        elif mode == "NUMBER":

            # CASE 1: ONE HAND (0–5)
            if len(all_landmarks) == 63:
                combined = all_landmarks + all_landmarks

            # CASE 2: TWO HANDS (6–10)
            elif len(all_landmarks) == 126:
                combined = all_landmarks

            else:
                combined = None

            if combined is not None:

                data = np.array(combined).reshape(42, 3)

                base_x, base_y, base_z = data[0]

                normalized = []
                for x, y_val, z in data:
                    normalized.extend([x - base_x, y_val - base_y, z - base_z])

                normalized = np.array(normalized).reshape(1, -1)

                pred = num_model.predict(normalized)
                current_pred = num_encoder.inverse_transform(pred)[0]

            else:
                current_pred = ""

        else:
            current_pred = ""

        # ==========================
        # 🔥 STABILITY (IMPROVED)
        # ==========================
        if current_pred == prev_prediction and current_pred != "":
            counter += 1
        else:
            counter = 0

        prev_prediction = current_pred

        if counter >= 6:
            prediction = current_pred

    # ==========================
    # DISPLAY
    # ==========================
    cv2.putText(img, f"Mode: {mode}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.putText(img, f"Output: {prediction}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

    cv2.imshow("GestureBridge System", img)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()