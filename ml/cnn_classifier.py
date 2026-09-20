# ============================================================
#  CNN-BASED GARBAGE TYPE CLASSIFICATION
# ============================================================
# Classifies an uploaded garbage image into one of 6 real classes
# from the trained dataset (TrashNet-style):
#     cardboard, glass, metal, paper, plastic, trash
#
# HOW IT WORKS (two modes):
#
#   MODE 1 - Real CNN (deep learning)  <-- used by default in this project
#       A CNN has been trained on ~2,500 real labelled garbage images
#       (see data/train/<class_name>/) and saved to:
#           models/cnn_garbage_classifier.h5
#       On import, this file loads that model with TensorFlow/Keras and
#       uses it to predict the class of any uploaded image.
#       To retrain it yourself (e.g. after adding more images), run:
#           python ml/train_cnn.py
#
#   MODE 2 - Fallback classifier (safety net)
#       If for any reason the trained model file is missing (e.g. you
#       haven't run train_cnn.py yet, or TensorFlow isn't installed),
#       we fall back to a simple, explainable colour-based rule
#       classifier so the app still runs end-to-end for a live demo.
#       This is clearly labelled in the UI as "fallback mode".
# ============================================================

import os
import numpy as np
from PIL import Image

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE = (96, 96)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "cnn_garbage_classifier.h5")

_cnn_model = None
_cnn_available = False
_tried_loading = False


def _try_load_cnn():
    """Attempts to load the trained CNN model. Silently falls back if unavailable."""
    global _cnn_model, _cnn_available, _tried_loading
    if _tried_loading:
        return
    _tried_loading = True
    try:
        from tensorflow.keras.models import load_model  # imported lazily (heavy library)
        if os.path.exists(MODEL_PATH):
            _cnn_model = load_model(MODEL_PATH)
            _cnn_available = True
    except Exception:
        _cnn_available = False


def build_cnn_model(num_classes=len(CLASSES)):
    """
    Defines the CNN architecture:
        [Conv2D -> MaxPool] x3 -> Flatten -> Dense -> Dropout -> Output

    Kept intentionally small and simple so it is easy to explain to an
    evaluator, and so it trains in a reasonable time on a laptop CPU.
    """
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def _fallback_classify(image_path):
    """
    Simple, explainable fallback classifier - NOT deep learning.
    Used automatically only if the trained CNN file is missing.
    Looks at average colour + brightness and applies common-sense rules.
    """
    img = Image.open(image_path).convert("RGB").resize((100, 100))
    arr = np.array(img).astype(float)
    r, g, b = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
    brightness = (r + g + b) / 3

    if brightness > 180 and abs(r - g) < 15 and abs(g - b) < 15:
        return "paper"        # bright, near-grey/white -> paper
    elif brightness > 140 and r > g and g > b:
        return "cardboard"    # warm light-brown tones -> cardboard
    elif b >= r and b >= g:
        return "plastic"      # cool/blue tones -> common plastic packaging
    elif brightness < 90:
        return "metal"        # dark, low reflectance -> approximated as metal
    elif r > 140 and g < 130 and b < 120:
        return "glass"        # amber/green glass tones
    else:
        return "trash"


def classify_garbage(image_path):
    """
    Main entry point used by the Streamlit app.

    Returns a tuple: (predicted_class: str, mode: str, confidence: float or None)
        mode is "CNN"      -> came from the trained deep learning model
        mode is "fallback" -> came from the simple rule-based classifier
        confidence is the model's probability for the predicted class (CNN mode only)
    """
    _try_load_cnn()

    if _cnn_available:
        img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        preds = _cnn_model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])
        return CLASSES[idx], "CNN", confidence
    else:
        return _fallback_classify(image_path), "fallback", None
