import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# ========== LOAD DATA ==========
DATA_PATH = "data_words"

X = []
y = []

for label in os.listdir(DATA_PATH):
    gesture_path = os.path.join(DATA_PATH, label)

    if os.path.isdir(gesture_path):
        for file in os.listdir(gesture_path):
            file_path = os.path.join(gesture_path, file)

            data = np.load(file_path)

            # 🔥 EXPECTING 126 FEATURES
            data = data.reshape(42, 3)

            # ==========================
            # 🔥 NORMALIZATION
            # ==========================
            base_x, base_y, base_z = data[0]

            normalized = []
            for x, y_val, z in data:
                normalized.extend([x - base_x, y_val - base_y, z - base_z])

            X.append(normalized)
            y.append(label)

X = np.array(X)
y = np.array(y)

# ==========================
# 🔥 ENCODE LABELS
# ==========================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ==========================
# 🔥 TRAIN TEST SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

# ==========================
# 🔥 MODEL
# ==========================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# 🔥 EVALUATE
# ==========================
accuracy = model.score(X_test, y_test)
print(f"Word Model Accuracy: {accuracy * 100:.2f}%")

# ==========================
# 🔥 SAVE
# ==========================
joblib.dump(model, "model_words.pkl")
joblib.dump(le, "label_encoder_words.pkl")

print("Word model saved successfully")