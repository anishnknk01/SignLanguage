# Indian Sign Language (ISL) Recognition & Sentence Formation

Real-time Indian Sign Language (ISL) static hand-sign recognition and sentence construction using a 1D Convolutional Neural Network (CNN) and MediaPipe HandLandmarker.

## Overview

The system tracks up to 2 hands in real time from a webcam stream, normalizes the 21 2D hand landmarks per hand, and classifies 36 static ISL signs (`0-9` and `A-Z`). For two-handed ISL signs where both hands touch or interact, predictions from both hands are fused into a single combined sign.

Recognized characters feed into a state-based sentence builder that requires consecutive frame stability and hand release / pose transition before accepting the next character, accompanied by n-gram word suggestions.

## Project Structure

- `gui_local.py` — CustomTkinter desktop interface with live camera feed, sentence builder, and word suggestions
- `recognition_engine.py` — MediaPipe HandLandmarker (VIDEO mode) wrapper and 1D CNN inference engine
- `sentence_builder.py` — Debounce and hand-release state machine for assembling characters into words and sentences
- `nlp_suggestions.py` — Trigram and bigram next-word suggestion engine
- `detect_cnn.py` — Entrypoint script (launches the desktop GUI by default, or standalone OpenCV HUD with `--cv2`)
- `train_cnn.py` — 1D CNN training script with landmark scale and noise augmentation
- `evaluate_cnn.py` — Evaluation script reporting accuracy, precision, recall, F1, and inference latency
- `model_cnn.h5` — Trained 1D CNN weights
- `classes_cnn.pkl` & `class_mapping.txt` — Class index to label mappings (`0-9`, `A-Z`)
- `hand_landmarker.task` — MediaPipe hand landmarker model bundle
- `dataset.pickle` — Extracted landmark dataset used for training and evaluation
- `Sign_dataset2/` — Raw image dataset organized by class (`0-9`, `A-Z`)

## Setup

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Run the Desktop GUI

```bash
python3 gui_local.py
```

**Keyboard Shortcuts:**
- `Space` / `Enter` — Complete current word
- `Backspace` — Delete last character or pull previous word back for editing
- `Esc` — Clear current word (or full sentence if word is empty)
- `1`–`5` — Select suggested word 1–5

### 2. Run OpenCV Window Only

```bash
python3 detect_cnn.py --cv2
```
Press `q` to quit or `d` to toggle top-5 prediction probabilities.

### 3. Train & Evaluate

```bash
python3 train_cnn.py
python3 evaluate_cnn.py
```

## License

MIT License
