"""VetVision training script.

This module trains an EfficientNetB0-based classifier using ImageDataGenerator
with proper EfficientNet preprocessing. Images are loaded from class-organized
folders created from labels.csv.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow import keras

# IDE uyumlu importlar
EfficientNetB0 = tf.keras.applications.EfficientNetB0
preprocess_input = tf.keras.applications.efficientnet.preprocess_input
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

# Hyper-parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
MODEL_PATH = "vetvision_model.h5"
LABELS_PATH = "labels.txt"
LABELS_CSV = Path("labels.csv")
SEED = 42

# Dataset paths
DATASET_ROOT = Path("dataset")
RAW_TRAIN_DIR = DATASET_ROOT / "train"
ORGANIZED_DIR = DATASET_ROOT / "organized"
TRAIN_DIR = ORGANIZED_DIR / "train"
VAL_DIR = ORGANIZED_DIR / "val"

VAL_SPLIT = 0.15


def organize_dataset() -> None:
    """Organize flat image folder into class subfolders for ImageDataGenerator."""
    if TRAIN_DIR.exists() and VAL_DIR.exists():
        # Check if already organized
        train_classes = list(TRAIN_DIR.iterdir())
        if train_classes and len(train_classes) > 10:
            print("Dataset already organized, skipping...")
            return

    print("Organizing dataset into class folders...")

    # Clean up existing organized folders
    if ORGANIZED_DIR.exists():
        shutil.rmtree(ORGANIZED_DIR)

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    # Read labels
    df = pd.read_csv(LABELS_CSV)
    print(f"Total images in labels.csv: {len(df)}")
    print(f"Total classes: {df['breed'].nunique()}")

    # Shuffle and split per class
    for breed in df["breed"].unique():
        breed_df = df[df["breed"] == breed].sample(frac=1.0, random_state=SEED)
        val_count = max(1, int(len(breed_df) * VAL_SPLIT))

        val_ids = breed_df.iloc[:val_count]["id"].tolist()
        train_ids = breed_df.iloc[val_count:]["id"].tolist()

        # Create class folders
        (TRAIN_DIR / breed).mkdir(exist_ok=True)
        (VAL_DIR / breed).mkdir(exist_ok=True)

        # Copy images
        for img_id in train_ids:
            src = RAW_TRAIN_DIR / f"{img_id}.jpg"
            dst = TRAIN_DIR / breed / f"{img_id}.jpg"
            if src.exists():
                shutil.copy2(src, dst)

        for img_id in val_ids:
            src = RAW_TRAIN_DIR / f"{img_id}.jpg"
            dst = VAL_DIR / breed / f"{img_id}.jpg"
            if src.exists():
                shutil.copy2(src, dst)

    print("Dataset organization complete!")


def create_generators():
    """Create training and validation data generators with EfficientNet preprocessing."""
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.15,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=SEED,
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    return train_generator, val_generator


def build_model(num_classes: int) -> keras.Model:
    """Build EfficientNetB0 model with frozen backbone."""
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
        pooling="avg",
    )

    # Freeze backbone for transfer learning
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation="softmax"),
    ], name="vetvision_efficientnet")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train() -> None:
    """Main training function."""
    print("=" * 60)
    print("VetVision Training Script")
    print("=" * 60)

    # Organize dataset
    organize_dataset()

    # Create data generators
    print("\nCreating data generators...")
    train_gen, val_gen = create_generators()

    # Print dataset info
    print("\n" + "=" * 60)
    print("DATASET INFORMATION:")
    print(f"  Training samples: {train_gen.samples}")
    print(f"  Validation samples: {val_gen.samples}")
    print(f"  Number of classes: {train_gen.num_classes}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Steps per epoch: {len(train_gen)}")
    print("=" * 60 + "\n")

    # Build model
    model = build_model(num_classes=train_gen.num_classes)
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # Train
    print("\nStarting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Save model and labels
    model.save(MODEL_PATH)

    # Save class indices
    class_to_idx = train_gen.class_indices
    with open(LABELS_PATH, "w", encoding="utf-8") as fp:
        json.dump(class_to_idx, fp, indent=2, ensure_ascii=False)

    # Print final results
    best_val_acc = max(history.history["val_accuracy"])
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    print(f"  Model saved to: {MODEL_PATH}")
    print(f"  Labels saved to: {LABELS_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.get_logger().setLevel("ERROR")
    train()
