#!/usr/bin/env python3
"""
Real-time Indian Sign Language (ISL) Hand-Sign Detector using CNN and MediaPipe.
"""

import os
import sys
import time
import pickle
import cv2
import numpy as np
from collections import deque, Counter
import mediapipe as mp
from tensorflow import keras

# Launch Desktop GUI with Sentence Builder by default
if __name__ == "__main__" and "--cv2" not in sys.argv and "--minimal" not in sys.argv:
    try:
        import gui_local
        gui_local.main()
        sys.exit(0)
    except Exception as e:
        print(f"Note: Falling back to direct OpenCV HUD mode ({e})...")

print("ISL Real-Time Hand-Sign Detector (CNN + MediaPipe)")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_cnn.h5")
CLASSES_PATH = os.path.join(BASE_DIR, "classes_cnn.pkl")
TASK_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

# Runtime Thresholds
CONFIDENCE_THRESHOLD = 0.72
SMOOTHING_WINDOW = 6
CONSECUTIVE_REQUIRED = 3

# MediaPipe Hand Skeleton Connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (9, 10), (10, 11), (11, 12),           # Middle
    (13, 14), (14, 15), (15, 16),          # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Knuckles
]

# 1. Load CNN Model
try:
    print("Loading CNN model...")
    model = keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH, "rb") as f:
        class_mapping = pickle.load(f)
    labels_dict = {int(k): str(v) for k, v in class_mapping.items()}
    print(f"CNN loaded ({len(labels_dict)} classes, input shape: {model.input_shape})")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# 2. Initialize MediaPipe HandLandmarker in VIDEO Tracking Mode
if not os.path.exists(TASK_PATH):
    print(f"Error: hand_landmarker.task not found at {TASK_PATH}")
    sys.exit(1)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=TASK_PATH),
    num_hands=2,
    min_hand_detection_confidence=0.28,
    min_hand_presence_confidence=0.28,
    min_tracking_confidence=0.28,
    running_mode=VisionRunningMode.VIDEO
)
hand_landmarker = HandLandmarker.create_from_options(options)
print("MediaPipe HandLandmarker initialized in VIDEO mode.")


class SignTracker:
    """Manages temporal voting, consecutive locking, and display metrics."""
    def __init__(self, name="Sign", primary_color=(0, 255, 0)):
        self.name = name
        self.primary_color = primary_color
        self.prediction_history = deque(maxlen=SMOOTHING_WINDOW)
        self.consecutive_char = None
        self.consecutive_count = 0
        self.active_sign = "-"
        self.active_confidence = 0.0
        self.raw_sign = "?"
        self.raw_confidence = 0.0
        self.status_text = "Sign not recognized"
        self.box_color = (0, 165, 255)
        self.predictions = None
        self.is_detected = False

    def reset(self):
        self.prediction_history.clear()
        self.consecutive_char = None
        self.consecutive_count = 0
        self.active_sign = "-"
        self.active_confidence = 0.0
        self.raw_sign = "?"
        self.raw_confidence = 0.0
        self.status_text = "Sign not recognized"
        self.box_color = (0, 165, 255)
        self.predictions = None
        self.is_detected = False

    def update(self, raw_sign, raw_confidence, predictions=None, prefix=""):
        self.is_detected = True
        self.predictions = predictions
        self.raw_sign = raw_sign
        self.raw_confidence = raw_confidence

        if raw_confidence >= CONFIDENCE_THRESHOLD:
            self.prediction_history.append(raw_sign)
            counts = Counter(self.prediction_history)
            stable_candidate, _ = counts.most_common(1)[0]

            if stable_candidate == self.consecutive_char:
                self.consecutive_count += 1
            else:
                self.consecutive_char = stable_candidate
                self.consecutive_count = 1

            if self.consecutive_count >= CONSECUTIVE_REQUIRED:
                self.active_sign = stable_candidate
                self.active_confidence = raw_confidence

            tag = f"{prefix}: " if prefix else ""
            self.status_text = f"{tag}{self.active_sign} ({self.active_confidence * 100:.0f}%)"
            self.box_color = self.primary_color
        else:
            self.prediction_history.clear()
            self.consecutive_count = 0
            tag = f"{prefix}: " if prefix else ""
            self.status_text = f"{tag}Sign not recognized"
            self.box_color = (0, 165, 255)


combined_tracker = SignTracker("Combined", (0, 255, 0))
left_tracker = SignTracker("Left", (0, 165, 255))
right_tracker = SignTracker("Right", (0, 255, 0))

# 3. Initialize Webcam
print("Opening webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera ready: {w}x{h}")
print("Controls: 'q' to quit, 'd' to toggle debug mode")

debug_mode = False
fps_history = deque(maxlen=30)
prev_time = time.time()
last_timestamp_ms = 0

was_combined_last_frame = False
touch_grace_counter = 0


def extract_features(hand_lms):
    """Normalize landmark coordinates invariant to distance and position."""
    xs = [lm.x for lm in hand_lms]
    ys = [lm.y for lm in hand_lms]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    denom_x = max_x - min_x + 1e-6
    denom_y = max_y - min_y + 1e-6
    norm = [[(lm.x - min_x) / denom_x, (lm.y - min_y) / denom_y] for lm in hand_lms]
    return np.array([norm], dtype='float32')


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    now = time.time()
    dt = now - prev_time
    prev_time = now
    if dt > 0:
        fps_history.append(1.0 / dt)

    timestamp_ms = int(now * 1000)
    if timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1
    last_timestamp_ms = timestamp_ms

    H, W, _ = frame.shape
    display_frame = frame.copy()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    detected_landmarks = detection_result.hand_landmarks or []
    num_hands = len(detected_landmarks)

    if num_hands == 2:
        lms0 = detected_landmarks[0]
        lms1 = detected_landmarks[1]

        pts0 = [(int(lm.x * W), int(lm.y * H)) for lm in lms0]
        pts1 = [(int(lm.x * W), int(lm.y * H)) for lm in lms1]
        pts0_arr = np.array(pts0)
        pts1_arr = np.array(pts1)

        x0_min, x0_max = pts0_arr[:, 0].min(), pts0_arr[:, 0].max()
        x1_min, x1_max = pts1_arr[:, 0].min(), pts1_arr[:, 0].max()
        gap_x = max(0, max(x0_min, x1_min) - min(x0_max, x1_max))

        dists = np.linalg.norm(pts0_arr[:, None, :] - pts1_arr[None, :, :], axis=2)
        min_landmark_dist = np.min(dists)

        hands_together = (gap_x < 110 or min_landmark_dist < 140)

        input_tensor_0 = extract_features(lms0)
        input_tensor_1 = extract_features(lms1)
        preds_0 = model.predict(input_tensor_0, verbose=0)[0]
        preds_1 = model.predict(input_tensor_1, verbose=0)[0]

        for idx1, idx2 in HAND_CONNECTIONS:
            cv2.line(display_frame, pts0[idx1], pts0[idx2], (0, 220, 255), 2, cv2.LINE_AA)
            cv2.line(display_frame, pts1[idx1], pts1[idx2], (255, 180, 0), 2, cv2.LINE_AA)
        for i, pt in enumerate(pts0):
            c = (0, 0, 255) if i in (4, 8, 12, 16, 20) else (0, 255, 0)
            cv2.circle(display_frame, pt, 4, c, -1, cv2.LINE_AA)
        for i, pt in enumerate(pts1):
            c = (0, 0, 255) if i in (4, 8, 12, 16, 20) else (0, 255, 0)
            cv2.circle(display_frame, pt, 4, c, -1, cv2.LINE_AA)

        if hands_together:
            was_combined_last_frame = True
            touch_grace_counter = 4
            left_tracker.reset()
            right_tracker.reset()

            p_combined = np.maximum(preds_0, preds_1)
            c_idx = int(np.argmax(p_combined))
            c_conf = float(p_combined[c_idx])
            c_sign = labels_dict.get(c_idx, "?")

            combined_tracker.update(c_sign, c_conf, predictions=p_combined, prefix="Combined")

            all_pts = np.vstack([pts0_arr, pts1_arr])
            bx1 = max(0, all_pts[:, 0].min() - 25)
            by1 = max(0, all_pts[:, 1].min() - 25)
            bx2 = min(W, all_pts[:, 0].max() + 25)
            by2 = min(H, all_pts[:, 1].max() + 25)

            cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), combined_tracker.box_color, 2)

            label_y = by1 - 12 if by1 > 35 else by2 + 25
            (tw, th), _ = cv2.getTextSize(combined_tracker.status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.rectangle(display_frame, (bx1, label_y - th - 6), (bx1 + tw + 10, label_y + 6), (15, 15, 25), -1)
            cv2.putText(display_frame, combined_tracker.status_text, (bx1 + 5, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, combined_tracker.box_color, 2)

            bottom_text = f"Hands: 2 Together | Combined Sign: {combined_tracker.active_sign} ({combined_tracker.active_confidence * 100:.0f}%)"
            bottom_color = (0, 255, 200)

        else:
            was_combined_last_frame = False
            touch_grace_counter = 0
            combined_tracker.reset()

            if pts0_arr[:, 0].mean() < pts1_arr[:, 0].mean():
                l_pts, r_pts = pts0_arr, pts1_arr
                l_preds, r_preds = preds_0, preds_1
            else:
                l_pts, r_pts = pts1_arr, pts0_arr
                l_preds, r_preds = preds_1, preds_0

            l_idx = int(np.argmax(l_preds))
            left_tracker.update(labels_dict.get(l_idx, "?"), float(l_preds[l_idx]), l_preds, prefix="Left")
            lx1 = max(0, l_pts[:, 0].min() - 20)
            ly1 = max(0, l_pts[:, 1].min() - 20)
            lx2 = min(W, l_pts[:, 0].max() + 20)
            ly2 = min(H, l_pts[:, 1].max() + 20)
            cv2.rectangle(display_frame, (lx1, ly1), (lx2, ly2), left_tracker.box_color, 2)
            lbl_ly = ly1 - 10 if ly1 > 30 else ly2 + 20
            cv2.putText(display_frame, left_tracker.status_text, (lx1, lbl_ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, left_tracker.box_color, 2)

            r_idx = int(np.argmax(r_preds))
            right_tracker.update(labels_dict.get(r_idx, "?"), float(r_preds[r_idx]), r_preds, prefix="Right")
            rx1 = max(0, r_pts[:, 0].min() - 20)
            ry1 = max(0, r_pts[:, 1].min() - 20)
            rx2 = min(W, r_pts[:, 0].max() + 20)
            ry2 = min(H, r_pts[:, 1].max() + 20)
            cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), right_tracker.box_color, 2)
            lbl_ry = ry1 - 10 if ry1 > 30 else ry2 + 20
            cv2.putText(display_frame, right_tracker.status_text, (rx1, lbl_ry),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, right_tracker.box_color, 2)

            bottom_text = f"Hands: 2 Apart | Left: {left_tracker.active_sign} | Right: {right_tracker.active_sign}"
            bottom_color = (255, 255, 255)

    elif num_hands == 1:
        hand_lms = detected_landmarks[0]
        pts = [(int(lm.x * W), int(lm.y * H)) for lm in hand_lms]
        pts_arr = np.array(pts)

        for idx1, idx2 in HAND_CONNECTIONS:
            cv2.line(display_frame, pts[idx1], pts[idx2], (255, 180, 0), 2, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            c = (0, 0, 255) if i in (4, 8, 12, 16, 20) else (0, 255, 0)
            cv2.circle(display_frame, pt, 4, c, -1, cv2.LINE_AA)

        input_tensor = extract_features(hand_lms)
        preds = model.predict(input_tensor, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        raw_conf = float(preds[pred_idx])
        raw_sign = labels_dict.get(pred_idx, "?")

        if touch_grace_counter > 0 and was_combined_last_frame:
            touch_grace_counter -= 1
            combined_tracker.update(raw_sign, raw_conf, predictions=preds, prefix="Combined")

            bx1 = max(0, pts_arr[:, 0].min() - 35)
            by1 = max(0, pts_arr[:, 1].min() - 35)
            bx2 = min(W, pts_arr[:, 0].max() + 35)
            by2 = min(H, pts_arr[:, 1].max() + 35)
            cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), combined_tracker.box_color, 2)
            label_y = by1 - 12 if by1 > 35 else by2 + 25
            (tw, th), _ = cv2.getTextSize(combined_tracker.status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.rectangle(display_frame, (bx1, label_y - th - 6), (bx1 + tw + 10, label_y + 6), (15, 15, 25), -1)
            cv2.putText(display_frame, combined_tracker.status_text, (bx1 + 5, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, combined_tracker.box_color, 2)

            bottom_text = f"Hands: 2 Interacting | Sign: {combined_tracker.active_sign} ({combined_tracker.active_confidence * 100:.0f}%)"
            bottom_color = (0, 255, 200)

        else:
            was_combined_last_frame = False
            touch_grace_counter = 0
            combined_tracker.reset()
            left_tracker.reset()

            right_tracker.update(raw_sign, raw_conf, preds, prefix="Sign")

            x1 = max(0, pts_arr[:, 0].min() - 25)
            y1 = max(0, pts_arr[:, 1].min() - 25)
            x2 = min(W, pts_arr[:, 0].max() + 25)
            y2 = min(H, pts_arr[:, 1].max() + 25)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), right_tracker.box_color, 2)

            label_y = y1 - 12 if y1 > 35 else y2 + 25
            (tw, th), _ = cv2.getTextSize(right_tracker.status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(display_frame, (x1, label_y - th - 6), (x1 + tw + 10, label_y + 6), (15, 15, 25), -1)
            cv2.putText(display_frame, right_tracker.status_text, (x1 + 5, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, right_tracker.box_color, 2)

            bottom_text = f"Hands: 1 detected | Prediction: {right_tracker.active_sign}"
            bottom_color = (255, 255, 255)

    else:
        was_combined_last_frame = False
        touch_grace_counter = 0
        combined_tracker.reset()
        left_tracker.reset()
        right_tracker.reset()

        bottom_text = "Hands: 0 detected | Show 1 or 2 hands to camera..."
        bottom_color = (130, 130, 255)

        cv2.putText(display_frame, "Show hand(s) to camera...",
                    (W // 2 - 160, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 255), 2)

    cv2.rectangle(display_frame, (0, H - 35), (W, H), (15, 15, 25), -1)
    cv2.putText(display_frame, bottom_text, (12, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bottom_color, 1)

    if debug_mode:
        active_preds = None
        panel_title = "Top Predictions"
        if combined_tracker.is_detected and combined_tracker.predictions is not None:
            active_preds = combined_tracker.predictions
            panel_title = "Combined Top-5:"
        elif right_tracker.is_detected and right_tracker.predictions is not None:
            active_preds = right_tracker.predictions
            panel_title = "Single Hand Top-5:"

        if active_preds is not None:
            top_5_idx = np.argsort(active_preds)[-5:][::-1]
            cv2.rectangle(display_frame, (10, 55), (195, 185), (15, 15, 25), -1)
            cv2.putText(display_frame, panel_title, (15, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 255), 1)
            for i, pidx in enumerate(top_5_idx):
                s = labels_dict.get(int(pidx), '?')
                c = active_preds[pidx] * 100
                cv2.putText(display_frame, f"#{i+1}: {s:<2} ({c:.1f}%)", (15, 95 + i * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imshow("ISL CNN Sign Detector", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        debug_mode = not debug_mode
        print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
hand_landmarker.close()
