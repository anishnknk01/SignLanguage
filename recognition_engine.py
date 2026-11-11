"""
ISL Recognition Engine.
Integrates MediaPipe HandLandmarker (VIDEO mode, num_hands=2) with the trained 1D CNN model.
Provides thread-safe real-time inference, two-hand combined gesture fusion, and HUD overlays.
"""

import os
import time
import pickle
import threading
from typing import Tuple, Optional
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_cnn.h5")
CLASSES_PATH = os.path.join(BASE_DIR, "classes_cnn.pkl")
TASK_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

# Skeleton connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (9, 10), (10, 11), (11, 12),           # Middle
    (13, 14), (14, 15), (15, 16),          # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Knuckles
]


class ProcessFrameResult(tuple):
    """
    Subclasses tuple to return (display_frame, predicted_sign, confidence, num_hands, status_text)
    for backward compatibility with unpackers, while also providing .landmarks attribute.
    """
    def __new__(cls, display_frame, predicted_sign, confidence, num_hands, status_text, landmarks=None):
        instance = super(ProcessFrameResult, cls).__new__(
            cls, (display_frame, predicted_sign, confidence, num_hands, status_text)
        )
        instance.landmarks = landmarks
        instance.display_frame = display_frame
        instance.predicted_sign = predicted_sign
        instance.confidence = confidence
        instance.num_hands = num_hands
        instance.status_text = status_text
        return instance


class ISLRecognitionEngine:
    """Thread-safe engine for MediaPipe tracking and ISL CNN classification."""

    def __init__(self, confidence_threshold: float = 0.70):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.labels_dict = {}
        self.hand_landmarker = None
        self.is_ready = False
        self.last_timestamp_ms = 0
        self.lock = threading.Lock()
        self.latest_landmarks = []

        # Performance tracking
        self.fps = 0.0
        self.prev_time = time.time()
        self.frame_count = 0

        # Occlusion grace period
        self.touch_grace_counter = 0
        self.was_combined_last_frame = False
        self.last_combined_sign = None
        self.last_combined_conf = 0.0

        self._load_resources()

    def _load_resources(self):
        """Loads CNN weights, class mapping, and MediaPipe landmarker."""
        print("Loading CNN model and MediaPipe task...")
        try:
            self.model = keras.models.load_model(MODEL_PATH)
            with open(CLASSES_PATH, "rb") as f:
                class_mapping = pickle.load(f)
            self.labels_dict = {int(k): str(v) for k, v in class_mapping.items()}
            print(f"CNN loaded ({len(self.labels_dict)} classes, input: {self.model.input_shape})")

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
            self.hand_landmarker = HandLandmarker.create_from_options(options)
            self.is_ready = True
            print("MediaPipe HandLandmarker initialized in VIDEO mode.")
        except Exception as e:
            print(f"Initialization error: {e}")
            self.is_ready = False

    def extract_features(self, hand_lms) -> np.ndarray:
        """Normalizes landmark coordinates relative to bounding box scale."""
        xs = [lm.x for lm in hand_lms]
        ys = [lm.y for lm in hand_lms]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        denom_x = max_x - min_x + 1e-6
        denom_y = max_y - min_y + 1e-6
        norm = [[(lm.x - min_x) / denom_x, (lm.y - min_y) / denom_y] for lm in hand_lms]
        return np.array([norm], dtype='float32')

    @staticmethod
    def normalize_landmarks_for_pose(hand_lms) -> np.ndarray:
        """
        Normalizes hand landmarks relative to the wrist and palm scale for pose comparison.
        Input: list of 21 landmark objects (each with .x, .y) or (21, 2) array.
        Returns: (21, 2) float32 numpy array.
        """
        if hasattr(hand_lms[0], 'x'):
            pts = np.array([[lm.x, lm.y] for lm in hand_lms], dtype=np.float32)
        else:
            pts = np.array(hand_lms, dtype=np.float32).reshape(21, 2)
        wrist = pts[0]
        centered = pts - wrist
        palm_size = float(np.linalg.norm(centered[9]))
        if palm_size < 1e-4:
            palm_size = float(np.max(np.linalg.norm(centered, axis=1))) + 1e-6
        return centered / palm_size

    def process_frame(
        self,
        frame: np.ndarray,
        flip: bool = True
    ) -> Tuple[np.ndarray, Optional[str], float, int, str]:
        """
        Process a single image frame.
        Returns:
            annotated_frame: np.ndarray (BGR)
            predicted_sign: str or None
            confidence: float (0.0 to 1.0)
            num_hands: int
            status_text: str
        """
        if not self.is_ready:
            return frame, None, 0.0, 0, "Model not initialized"

        with self.lock:
            now = time.time()
            dt = now - self.prev_time
            self.prev_time = now
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps > 0 else 1.0 / dt

            if flip:
                frame = cv2.flip(frame, 1)

            H, W, _ = frame.shape
            display_frame = frame.copy()

            timestamp_ms = int(now * 1000)
            if timestamp_ms <= self.last_timestamp_ms:
                timestamp_ms = self.last_timestamp_ms + 1
            self.last_timestamp_ms = timestamp_ms

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            detected_landmarks = detection_result.hand_landmarks or []
            num_hands = len(detected_landmarks)

            normalized_landmarks = [self.normalize_landmarks_for_pose(h) for h in detected_landmarks]
            self.latest_landmarks = normalized_landmarks

            predicted_sign = None
            confidence = 0.0
            status_text = "Show hand(s) to camera..."

            if num_hands == 2:
                lms0, lms1 = detected_landmarks[0], detected_landmarks[1]
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

                preds_0 = self.model.predict(self.extract_features(lms0), verbose=0)[0]
                preds_1 = self.model.predict(self.extract_features(lms1), verbose=0)[0]

                for idx1, idx2 in HAND_CONNECTIONS:
                    cv2.line(display_frame, pts0[idx1], pts0[idx2], (0, 220, 255), 2, cv2.LINE_AA)
                    cv2.line(display_frame, pts1[idx1], pts1[idx2], (255, 180, 0), 2, cv2.LINE_AA)
                for pt in pts0:
                    cv2.circle(display_frame, pt, 4, (0, 255, 0), -1, cv2.LINE_AA)
                for pt in pts1:
                    cv2.circle(display_frame, pt, 4, (0, 255, 0), -1, cv2.LINE_AA)

                if hands_together:
                    self.was_combined_last_frame = True
                    self.touch_grace_counter = 4

                    p_combined = np.maximum(preds_0, preds_1)
                    c_idx = int(np.argmax(p_combined))
                    confidence = float(p_combined[c_idx])
                    raw_sign = self.labels_dict.get(c_idx, "?")

                    all_pts = np.vstack([pts0_arr, pts1_arr])
                    bx1 = max(0, all_pts[:, 0].min() - 25)
                    by1 = max(0, all_pts[:, 1].min() - 25)
                    bx2 = min(W, all_pts[:, 0].max() + 25)
                    by2 = min(H, all_pts[:, 1].max() + 25)

                    if confidence >= self.confidence_threshold:
                        predicted_sign = raw_sign
                        box_color = (0, 255, 0)
                        status_text = f"Sign: {raw_sign} ({confidence * 100:.0f}%)"
                    else:
                        box_color = (0, 165, 255)
                        status_text = "Sign uncertain"

                    self.last_combined_sign = predicted_sign
                    self.last_combined_conf = confidence

                    cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), box_color, 2)
                    label_y = by1 - 10 if by1 > 30 else by2 + 25
                    (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(display_frame, (bx1, label_y - th - 6), (bx1 + tw + 10, label_y + 6), (15, 15, 25), -1)
                    cv2.putText(display_frame, status_text, (bx1 + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

                else:
                    self.was_combined_last_frame = False
                    self.touch_grace_counter = 0

                    p0_idx = int(np.argmax(preds_0))
                    p1_idx = int(np.argmax(preds_1))
                    conf0 = float(preds_0[p0_idx])
                    conf1 = float(preds_1[p1_idx])

                    if conf0 >= conf1 and conf0 >= self.confidence_threshold:
                        predicted_sign = self.labels_dict.get(p0_idx)
                        confidence = conf0
                    elif conf1 >= self.confidence_threshold:
                        predicted_sign = self.labels_dict.get(p1_idx)
                        confidence = conf1

                    for pts_arr, conf_val, idx_val in [(pts0_arr, conf0, p0_idx), (pts1_arr, conf1, p1_idx)]:
                        lx1 = max(0, pts_arr[:, 0].min() - 20)
                        ly1 = max(0, pts_arr[:, 1].min() - 20)
                        lx2 = min(W, pts_arr[:, 0].max() + 20)
                        ly2 = min(H, pts_arr[:, 1].max() + 20)
                        color = (0, 255, 0) if conf_val >= self.confidence_threshold else (0, 165, 255)
                        s = self.labels_dict.get(idx_val, "?")
                        cv2.rectangle(display_frame, (lx1, ly1), (lx2, ly2), color, 2)
                        cv2.putText(display_frame, f"{s} ({conf_val * 100:.0f}%)", (lx1, ly1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    status_text = "2 Hands Apart"

            elif num_hands == 1:
                lms = detected_landmarks[0]
                pts = [(int(lm.x * W), int(lm.y * H)) for lm in lms]
                pts_arr = np.array(pts)

                for idx1, idx2 in HAND_CONNECTIONS:
                    cv2.line(display_frame, pts[idx1], pts[idx2], (255, 180, 0), 2, cv2.LINE_AA)
                for pt in pts:
                    cv2.circle(display_frame, pt, 4, (0, 255, 0), -1, cv2.LINE_AA)

                preds = self.model.predict(self.extract_features(lms), verbose=0)[0]
                p_idx = int(np.argmax(preds))
                confidence = float(preds[p_idx])
                raw_sign = self.labels_dict.get(p_idx, "?")

                if self.touch_grace_counter > 0 and self.was_combined_last_frame:
                    self.touch_grace_counter -= 1
                    predicted_sign = self.last_combined_sign or raw_sign
                    confidence = max(confidence, self.last_combined_conf)
                    status_text = f"Combined Sign: {predicted_sign} ({confidence * 100:.0f}%)"
                    box_color = (0, 255, 0)
                else:
                    self.was_combined_last_frame = False
                    self.touch_grace_counter = 0
                    if confidence >= self.confidence_threshold:
                        predicted_sign = raw_sign
                        box_color = (0, 255, 0)
                        status_text = f"Sign: {raw_sign} ({confidence * 100:.0f}%)"
                    else:
                        box_color = (0, 165, 255)
                        status_text = "Sign uncertain"

                x1 = max(0, pts_arr[:, 0].min() - 25)
                y1 = max(0, pts_arr[:, 1].min() - 25)
                x2 = min(W, pts_arr[:, 0].max() + 25)
                y2 = min(H, pts_arr[:, 1].max() + 25)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(display_frame, status_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

            else:
                self.was_combined_last_frame = False
                self.touch_grace_counter = 0
                status_text = "Show hand(s) to camera..."

            return ProcessFrameResult(
                display_frame, predicted_sign, confidence, num_hands, status_text, normalized_landmarks
            )
