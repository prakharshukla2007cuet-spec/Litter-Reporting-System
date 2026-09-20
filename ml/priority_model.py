# ============================================================
#  PRIORITY PREDICTION using RANDOM FOREST
# ============================================================
# Predicts how urgently a reported litter spot needs cleaning:
#       "High"  /  "Medium"  /  "Low"
#
# FEATURES USED:
#   - garbage_type      : some waste types are more hazardous (glass, metal)
#   - hazard_score       : numeric hazard rating derived from garbage_type
#   - duplicate_count    : how many times this SAME spot has been reported
#                          (more repeat reports = bigger, more urgent problem)
#   - gps_accuracy        : GPS accuracy in meters (lower = more trustworthy report)
#
# The model is a scikit-learn RandomForestClassifier trained on a small
# rule-based synthetic dataset (see train_priority_model.py) so the logic
# stays simple and explainable to an evaluator:
#   "hazardous type + many repeat reports => High priority"
# A ready-trained model is shipped in models/priority_rf_model.pkl so the
# app works immediately without needing to retrain anything.
# ============================================================

import os
import joblib
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "priority_rf_model.pkl")

# must match ml/cnn_classifier.py CLASSES exactly (same order, trained dataset classes)
GARBAGE_TYPES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
HAZARD_SCORE = {"glass": 3, "metal": 3, "plastic": 2, "trash": 2, "cardboard": 1, "paper": 1}

_model = None


def _load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def _build_features(garbage_type, duplicate_count, accuracy):
    hazard = HAZARD_SCORE.get(garbage_type, 1)
    type_idx = GARBAGE_TYPES.index(garbage_type) if garbage_type in GARBAGE_TYPES else len(GARBAGE_TYPES) - 1
    acc = accuracy if accuracy else 50
    return np.array([[type_idx, hazard, duplicate_count, acc]])


def predict_priority(garbage_type, duplicate_count=1, accuracy=50):
    """Returns 'High', 'Medium', or 'Low'."""
    model = _load_model()
    X = _build_features(garbage_type, duplicate_count, accuracy)
    return model.predict(X)[0]
