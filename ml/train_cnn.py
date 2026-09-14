# ============================================================
#  TRAIN THE CNN GARBAGE CLASSIFIER
# ============================================================
# This is OPTIONAL. The app already works using a fallback
# classifier (see cnn_classifier.py) if you skip this step.
#
# To train the real CNN:
#   1) Install TensorFlow:      pip install tensorflow
#   2) Add labelled images to:
#         data/train/plastic/*.jpg
#         data/train/paper/*.jpg
#         data/train/organic/*.jpg
#         data/train/metal/*.jpg
#         data/train/glass/*.jpg
#         data/train/other/*.jpg
#      (A free dataset like "TrashNet" or "Garbage Classification"
#       on Kaggle works well for this.)
#   3) Run:  python ml/train_cnn.py
#
# The trained model is saved to models/cnn_garbage_classifier.h5
# and will automatically be picked up by the app on next run.
# ============================================================

import os
import sys

sys.path.append(os.path.dirname(__file__))
from cnn_classifier import build_cnn_model, CLASSES, IMG_SIZE, MODEL_PATH

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "train")


def main():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    if not os.path.exists(DATA_DIR):
        print(f"No training data folder found at: {DATA_DIR}")
        print("Create sub-folders per class and add images before training.")
        return

    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        DATA_DIR, target_size=IMG_SIZE, batch_size=16,
        class_mode="categorical", subset="training", classes=CLASSES
    )
    val_gen = datagen.flow_from_directory(
        DATA_DIR, target_size=IMG_SIZE, batch_size=16,
        class_mode="categorical", subset="validation", classes=CLASSES
    )

    if train_gen.samples == 0:
        print("No images found inside data/train/<class_name>/ folders.")
        print("Add some images first, then re-run this script.")
        return

    model = build_cnn_model()
    model.fit(train_gen, validation_data=val_gen, epochs=10)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\n✅ Model trained and saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
