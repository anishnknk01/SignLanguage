#!/usr/bin/env python3
"""
Evaluation and Metrics Benchmark Script for ISL CNN Model.
Reports Test Accuracy, Per-Class Precision/Recall/F1, Confusion Matrix,
and CPU Inference Latency.
"""

import os
import sys
import time
import pickle
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model_cnn.h5")
CLASSES_PATH = os.path.join(PROJECT_ROOT, "models", "classes_cnn.pkl")
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "dataset.pickle")


def evaluate():
    print("ISL CNN Model Evaluation & Benchmark")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATASET_PATH):
        print(f"Error: Model or dataset not found at {MODEL_PATH}")
        sys.exit(1)

    model = keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH, "rb") as f:
        class_mapping = pickle.load(f)
    labels_dict = {int(k): str(v) for k, v in class_mapping.items()}

    with open(DATASET_PATH, "rb") as f:
        data_dict = pickle.load(f)

    raw_X = np.asarray(data_dict['data'])
    raw_y = np.asarray([int(l) for l in data_dict['labels']])

    X_cnn = raw_X.reshape(-1, 21, 2)
    x_train, x_test, y_train, y_test = train_test_split(
        X_cnn, raw_y, test_size=0.20, random_state=42, stratify=raw_y
    )

    print(f"Model loaded: {len(labels_dict)} classes, input shape: {model.input_shape}")
    print(f"Held-out Test Set: {len(x_test):,} samples")

    t_start = time.time()
    preds_prob = model.predict(x_test, verbose=0)
    total_time = time.time() - t_start

    y_pred = np.argmax(preds_prob, axis=1)

    acc = accuracy_score(y_test, y_pred) * 100.0
    macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0) * 100.0
    macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0) * 100.0
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0) * 100.0

    latency_ms = (total_time / len(y_test)) * 1000.0
    approx_fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

    print("\nTest Results:")
    print(f"  Test Accuracy:          {acc:.2f}%")
    print(f"  Macro Precision:        {macro_prec:.2f}%")
    print(f"  Macro Recall:           {macro_rec:.2f}%")
    print(f"  Macro F1-Score:         {macro_f1:.2f}%")
    print(f"  CPU Inference Latency:  {latency_ms:.3f} ms / sample (~{approx_fps:.0f} FPS)\n")

    target_names = [labels_dict.get(i, str(i)) for i in range(len(labels_dict))]
    print("Per-Class Performance:")
    print(classification_report(y_test, y_pred, target_names=target_names, digits=3, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    confused = []
    for i in range(len(target_names)):
        for j in range(len(target_names)):
            if i != j and cm[i, j] > 0:
                confused.append((labels_dict[i], labels_dict[j], int(cm[i, j])))

    confused.sort(key=lambda x: x[2], reverse=True)
    if confused:
        print("\nCommonly Confused Sign Pairs:")
        for t_cls, p_cls, count in confused[:8]:
            print(f"  True '{t_cls}' -> Predicted '{p_cls}': {count} errors")
    else:
        print("\nNo misclassifications on test split.")


if __name__ == "__main__":
    evaluate()
