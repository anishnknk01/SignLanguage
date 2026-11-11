# Indian Sign Language (ISL) Recognition & Sentence Formation

Real-time Indian Sign Language (ISL) static hand-sign recognition and sentence construction using a 1D Convolutional Neural Network (CNN) and MediaPipe HandLandmarker.

## Overview

The system tracks up to 2 hands in real time from a webcam stream, normalizes the 21 2D hand landmarks per hand, and classifies 36 static ISL signs (`0-9` and `A-Z`). For two-handed ISL signs where both hands touch or interact, predictions from both hands are fused into a single combined sign.

Recognized characters feed into a state-based sentence builder that requires consecutive frame stability and hand release / pose transition before accepting the next character, accompanied by n-gram word suggestions.

## Project Structure

```text
SignLanguage/
├── main.py                      # Application entrypoint
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── src/
│   ├── gui_local.py             # CustomTkinter desktop interface
│   ├── recognition_engine.py    # MediaPipe HandLandmarker & 1D CNN inference engine
│   ├── sentence_builder.py      # Debounce and hand-release state machine
│   ├── nlp_suggestions.py       # Trigram and bigram next-word suggestion engine
│   ├── detect_cnn.py            # Standalone OpenCV HUD detector
│   ├── train_cnn.py             # 1D CNN training script with landmark augmentation
│   └── evaluate_cnn.py          # Evaluation and latency benchmark script
├── models/
│   ├── model_cnn.h5             # Trained 1D CNN weights
│   ├── classes_cnn.pkl          # Serialized class mapping (0-9, A-Z)
│   ├── class_mapping.txt        # Human-readable class index mapping
│   └── hand_landmarker.task     # MediaPipe hand landmarker model bundle
└── data/
    ├── dataset.pickle           # Extracted landmark dataset for training/evaluation
    └── Sign_dataset2/           # Raw image dataset organized by class (0-9, A-Z)
```

## Setup

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Run the Desktop GUI

```bash
python3 main.py
# or
python3 src/gui_local.py
```

**Keyboard Shortcuts:**
- `Space` / `Enter` — Complete current word
- `Backspace` — Delete last character or pull previous word back for editing
- `Esc` — Clear current word (or full sentence if word is empty)
- `1`–`5` — Select suggested word 1–5

### 2. Run OpenCV Window Only

```bash
python3 main.py --cv2
# or
python3 src/detect_cnn.py --cv2
```
Press `q` to quit or `d` to toggle top-5 prediction probabilities.

### 3. Train & Evaluate

```bash
python3 src/train_cnn.py
python3 src/evaluate_cnn.py
```

