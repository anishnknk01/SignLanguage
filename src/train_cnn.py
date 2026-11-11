#!/usr/bin/env python3
"""
ISL 1D CNN Training Pipeline.
Trains a sequence-convolutional neural network on normalized MediaPipe hand landmarks.
"""

import os
import time
import pickle
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("ISL Hand-Sign Recognition - 1D CNN Training")

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "dataset.pickle")
OUTPUT_MODEL = os.path.join(PROJECT_ROOT, "models", "model_cnn.h5")
OUTPUT_CLASSES = os.path.join(PROJECT_ROOT, "models", "classes_cnn.pkl")

EPOCHS = 15
BATCH_SIZE = 64


def load_and_augment_data():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")

    with open(DATASET_PATH, "rb") as f:
        data_dict = pickle.load(f)

    raw_X = np.asarray(data_dict['data'])
    raw_y = np.asarray([int(l) for l in data_dict['labels']])
    class_mapping = data_dict.get('class_mapping', {})

    print(f"Loaded {len(raw_X):,} base samples across {len(class_mapping)} classes")

    # Landmark Data Augmentation (Scale jitter + Gaussian perturbation)
    print("Augmenting landmark geometry (3x augmentation)...")
    X_aug = list(raw_X)
    y_aug = list(raw_y)
    rng = np.random.default_rng(42)

    for x_sample, y_sample in zip(raw_X, raw_y):
        for _ in range(2):
            scale = rng.uniform(0.97, 1.03)
            noise = rng.normal(0, 0.01, size=x_sample.shape)
            perturbed = np.clip(x_sample * scale + noise, 0.0, 1.0)
            X_aug.append(perturbed)
            y_aug.append(y_sample)

    X_aug = np.array(X_aug).reshape(-1, 21, 2)
    y_aug = np.array(y_aug)
    print(f"Augmented dataset size: {len(X_aug):,} samples")

    return X_aug, y_aug, class_mapping


def build_1d_cnn(num_classes):
    print("Constructing 1D CNN architecture...")
    model = keras.Sequential([
        layers.Input(shape=(21, 2)),
        layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),

        layers.Conv1D(256, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.3),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ], name="ISL_1D_CNN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    X, y, class_mapping = load_and_augment_data()
    num_classes = len(class_mapping)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    print(f"Training set: {len(x_train):,} | Test set: {len(x_test):,}")

    model = build_1d_cnn(num_classes)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(OUTPUT_MODEL, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    print("Training 1D CNN model...")
    start_t = time.time()
    model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    total_time = time.time() - start_t

    with open(OUTPUT_CLASSES, 'wb') as f:
        pickle.dump(class_mapping, f)
    print(f"Saved class mapping to {OUTPUT_CLASSES}")

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred) * 100.0

    print("\nEvaluation Results:")
    print(f"  Held-out Test Accuracy: {acc:.2f}%")
    print(f"  Training Duration:      {total_time:.1f}s")
    print(f"  Model File:             {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
