# ============================================================
#  TRAIN THE CNN GARBAGE CLASSIFIER
# ============================================================
# Trains a small CNN on the labelled images in data/train/<class>/
# (this project ships with a real dataset of ~2,500 garbage photos
# across 6 classes: cardboard, glass, metal, paper, plastic, trash).
#
# Run:
#   pip install tensorflow          (only needed for training)
#   python ml/train_cnn.py
#
# The trained model is saved to models/cnn_garbage_classifier.h5
# and is automatically picked up by ml/cnn_classifier.py the next
# time the app runs - no other code changes needed.
# ============================================================

import os
import sys

sys.path.append(os.path.dirname(__file__))
from cnn_classifier import build_cnn_model, CLASSES, IMG_SIZE, MODEL_PATH

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "train")
EPOCHS = 12
BATCH_SIZE = 32


def main():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping

    if not os.path.exists(DATA_DIR):
        print(f"No training data folder found at: {DATA_DIR}")
        print("Create sub-folders per class and add images before training.")
        return

    # Data augmentation on the training split only - helps the small CNN
    # generalise better since we only have a few hundred images per class.
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.15,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
    )

    train_gen = datagen.flow_from_directory(
        DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="training", classes=CLASSES
    )
    val_gen = datagen.flow_from_directory(
        DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="validation", classes=CLASSES, shuffle=False
    )

    if train_gen.samples == 0:
        print("No images found inside data/train/<class_name>/ folders.")
        print("Add some images first, then re-run this script.")
        return

    print(f"Found {train_gen.samples} training images and {val_gen.samples} validation images.")
    print(f"Classes: {CLASSES}")

    model = build_cnn_model()
    model.summary()

    early_stop = EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[early_stop],
    )

    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"\nFinal validation accuracy: {val_acc * 100:.2f}%")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"✅ Model trained and saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
