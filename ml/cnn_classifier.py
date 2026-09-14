# ============================================================
#  CNN-BASED GARBAGE TYPE CLASSIFICATION
# ============================================================
# Classifies an uploaded garbage image into one of:
#   plastic, paper, organic, metal, glass, other
#
# HOW IT WORKS (two modes):
#
#   MODE 1 - Real CNN (deep learning):
#       If a trained model file exists at models/cnn_garbage_classifier.h5
#       we load it with TensorFlow/Keras and use it to predict the class.
#       Train it yourself by running:  python ml/train_cnn.py
#       (after placing labelled images in data/train/<class_name>/)
#
#   MODE 2 - Fallback classifier (no training data needed):
#       If no trained model is found, we use a simple, easy-to-explain
#       color-based rule classifier so the app still works end-to-end
#       for a live demo. This is clearly labelled in the UI as
#       "fallback mode" so it stays honest for an evaluator.
#
# This two-mode design means the project ALWAYS runs, even before you
# have collected a labelled garbage-image dataset to train the CNN on.
# ============================================================

import os
import numpy as np
from PIL import Image

CLASSES = ["plastic", "paper", "organic", "metal", "glass", "other"]
IMG_SIZE = (64, 64)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "cnn_garbage_classifier.h5")

_cnn_model = None
_cnn_available = False
_tried_loading = False


def _try_load_cnn():
    """Attempts to load a trained CNN model. Silently falls back if unavailable."""
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
        Conv2D -> MaxPool -> Conv2D -> MaxPool -> Flatten -> Dense -> Output

    Kept intentionally small and simple so it is easy to explain to an
    evaluator, and so it trains quickly on a laptop CPU.
    """
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), activation="relu"),
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
    Looks at the average colour + brightness of the image and applies
    a few common-sense rules. Used automatically until a real CNN is trained.
    """
    img = Image.open(image_path).convert("RGB").resize((100, 100))
    arr = np.array(img).astype(float)
    r, g, b = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
    brightness = (r + g + b) / 3

    if g > r and g > b and g > 90:
        return "organic"     # green-ish tones -> plant / food waste
    elif brightness < 80:
        return "metal"       # dark, low reflectance -> approximated as metal
    elif brightness > 180 and abs(r - g) < 15 and abs(g - b) < 15:
        return "paper"       # bright, near-grey/white -> paper / cardboard
    elif b >= r and b >= g:
        return "plastic"     # cool/blue tones -> common plastic packaging
    elif r > 150 and g < 130 and b < 130:
        return "glass"       # warm amber/brown tones -> approximated as glass bottles
    else:
        return "other"


def classify_garbage(image_path):
    """
    Main entry point used by the Streamlit app.

    Returns a tuple: (predicted_class: str, mode: str)
        mode is "CNN"      -> came from the trained deep learning model
        mode is "fallback" -> came from the simple rule-based classifier
    """
    _try_load_cnn()

    if _cnn_available:
        img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        preds = _cnn_model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(preds))
        return CLASSES[idx], "CNN"
    else:
        return _fallback_classify(image_path), "fallback"
