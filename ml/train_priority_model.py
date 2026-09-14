# ============================================================
#  TRAIN THE PRIORITY PREDICTION MODEL (Random Forest)
# ============================================================
# Generates a small rule-based synthetic dataset and trains a
# RandomForestClassifier on it to predict report priority.
#
# A trained model is ALREADY included in models/priority_rf_model.pkl,
# so you only need to run this again if you want to change the rules
# or regenerate the model.
#
# Run:  python ml/train_priority_model.py
# ============================================================

import os
import random
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

GARBAGE_TYPES = ["plastic", "paper", "organic", "metal", "glass", "other"]
HAZARD_SCORE = {"glass": 3, "metal": 3, "plastic": 2, "other": 1, "paper": 1, "organic": 1}
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "priority_rf_model.pkl")


def label_priority(hazard, duplicate_count, accuracy):
    """The 'ground truth' rule used to generate training labels."""
    score = (hazard * 2) + (min(duplicate_count, 10) * 1.5) - (accuracy / 50)
    if score >= 9:
        return "High"
    elif score >= 5:
        return "Medium"
    else:
        return "Low"


def generate_dataset(n=2000):
    X, y = [], []
    for _ in range(n):
        gtype = random.choice(GARBAGE_TYPES)
        hazard = HAZARD_SCORE[gtype]
        dup = random.randint(1, 12)
        acc = random.randint(5, 150)
        type_idx = GARBAGE_TYPES.index(gtype)

        X.append([type_idx, hazard, dup, acc])
        y.append(label_priority(hazard, dup, acc))
    return np.array(X), np.array(y)


def main():
    X, y = generate_dataset()

    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"✅ Priority model trained and saved to: {MODEL_PATH}")
    print("Training accuracy:", round(model.score(X, y) * 100, 2), "%")


if __name__ == "__main__":
    main()
