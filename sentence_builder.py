"""
Sentence Builder State and Recognition Logic for Indian Sign Language (ISL).
Manages character-by-character formation with a 5-stage state machine:
  READY -> STABILIZING -> ACCEPTED -> WAITING_FOR_RELEASE -> READY.
"""

import time
from typing import Dict, Any, List, Optional
import numpy as np
from nlp_suggestions import WordSuggestionEngine


class BuilderState:
    READY = "READY"
    STABILIZING = "STABILIZING"
    ACCEPTED = "ACCEPTED"
    WAITING_FOR_RELEASE = "WAITING_FOR_RELEASE"


class SentenceBuilder:
    """
    Core state manager for ISL sign-to-text sentence construction.
    Implements a 5-state debounce and hand-release state machine.
    """

    def __init__(
        self,
        stable_frames_required: int = 10,
        hand_release_duration_sec: float = 0.25,
        landmark_diff_threshold: float = 0.10,
        min_confidence: float = 0.70,
        gesture_cooldown_sec: float = 0.0,
        same_sign_hold_sec: float = 0.0,
        **kwargs
    ):
        self.suggestion_engine = WordSuggestionEngine()

        # Text State
        self.current_word: str = ""
        self.completed_sentence: str = ""
        self.word_suggestions: List[str] = self.suggestion_engine.get_suggestions("")

        # Live Sign Tracking Preview
        self.live_sign: str = "-"
        self.live_confidence: float = 0.0
        self.is_hands_detected: bool = False
        self.hand_count: int = 0
        self.status_message: str = "Show hand(s) to camera..."

        # State Machine Configuration
        self.state: str = BuilderState.READY
        self.stable_frames_required: int = max(1, stable_frames_required)
        self.hand_release_duration_sec: float = hand_release_duration_sec
        self.landmark_diff_threshold: float = landmark_diff_threshold
        self.min_confidence: float = min_confidence

        # Stability Buffers
        self.consecutive_sign: Optional[str] = None
        self.consecutive_count: int = 0

        # Accepted Sign & Locked Pose State
        self.accepted_sign: Optional[str] = None
        self.accepted_landmarks: Optional[List[np.ndarray]] = None
        self.accepted_timestamp: float = 0.0

        # Hand Disappearance Tracking (Time-based release)
        self.hand_disappeared_start_time: Optional[float] = None

        # History and Undo Stack
        self.last_added_sign: Optional[str] = None
        self.last_addition_time: float = 0.0
        self.has_released_pose: bool = True
        self.recognition_history: List[Dict[str, Any]] = []
        self.undo_stack: List[Dict[str, Any]] = []
        self._history_id_counter = 0

    @staticmethod
    def compute_pose_distance(
        lms1: Optional[List[np.ndarray]],
        lms2: Optional[List[np.ndarray]]
    ) -> float:
        """
        Computes the normalized landmark Euclidean distance between two frames of hand landmarks.
        Handles single-hand, dual-hand permutations, and hand count changes.
        """
        if lms1 is None or lms2 is None or len(lms1) == 0 or len(lms2) == 0:
            return float('inf')

        if len(lms1) != len(lms2):
            return 1.0

        if len(lms1) == 1:
            try:
                h1 = np.asarray(lms1[0], dtype=np.float32).reshape(21, 2)
                h2 = np.asarray(lms2[0], dtype=np.float32).reshape(21, 2)
                return float(np.mean(np.linalg.norm(h1 - h2, axis=1)))
            except Exception:
                return 1.0

        if len(lms1) == 2 and len(lms2) == 2:
            try:
                h1_a = np.asarray(lms1[0], dtype=np.float32).reshape(21, 2)
                h1_b = np.asarray(lms1[1], dtype=np.float32).reshape(21, 2)
                h2_a = np.asarray(lms2[0], dtype=np.float32).reshape(21, 2)
                h2_b = np.asarray(lms2[1], dtype=np.float32).reshape(21, 2)
                d_direct = 0.5 * (
                    np.mean(np.linalg.norm(h1_a - h2_a, axis=1)) +
                    np.mean(np.linalg.norm(h1_b - h2_b, axis=1))
                )
                d_swapped = 0.5 * (
                    np.mean(np.linalg.norm(h1_a - h2_b, axis=1)) +
                    np.mean(np.linalg.norm(h1_b - h2_a, axis=1))
                )
                return float(min(d_direct, d_swapped))
            except Exception:
                return 1.0

        return 1.0

    def _push_undo_snapshot(self):
        """Save a snapshot of current text state to the undo stack."""
        snapshot = {
            "current_word": self.current_word,
            "completed_sentence": self.completed_sentence,
            "last_added_sign": self.last_added_sign,
        }
        self.undo_stack.append(snapshot)
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def _add_history_entry(self, entry_type: str, value: str, details: str):
        """Append an event to the recognition history."""
        self._history_id_counter += 1
        entry = {
            "id": self._history_id_counter,
            "type": entry_type,
            "value": value,
            "details": details,
            "timestamp": time.strftime("%H:%M:%S")
        }
        self.recognition_history.insert(0, entry)
        if len(self.recognition_history) > 30:
            self.recognition_history.pop()

    def update_live_detection(
        self,
        predicted_sign: Optional[str],
        confidence: float,
        num_hands: int,
        status_msg: str = "",
        landmarks: Optional[Any] = None
    ) -> Optional[str]:
        """
        Process a single vision frame's prediction using the 5-state machine:
          1. READY: Wait for a valid hand and candidate sign.
          2. STABILIZING: Collect consecutive predictions across required frames.
          3. ACCEPTED: Insert recognized character once and lock insertion.
          4. WAITING_FOR_RELEASE: Lock character insertion while same hand pose is held.
          5. READY AGAIN: Unlock recognition after gesture change or hand disappearance.
        """
        now = time.time()
        self.live_confidence = confidence
        self.hand_count = num_hands
        self.is_hands_detected = (num_hands > 0)

        norm_landmarks: Optional[List[np.ndarray]] = None
        if landmarks is not None:
            if isinstance(landmarks, list):
                norm_landmarks = landmarks
            elif isinstance(landmarks, np.ndarray):
                norm_landmarks = [landmarks]

        is_hand_present = (num_hands > 0)
        is_valid_sign = (
            is_hand_present and
            predicted_sign is not None and
            predicted_sign not in ("-", "—", "?", "") and
            confidence >= self.min_confidence
        )

        if is_valid_sign:
            self.live_sign = predicted_sign
        elif is_hand_present and predicted_sign and predicted_sign not in ("-", "—", ""):
            self.live_sign = predicted_sign
        else:
            self.live_sign = "-"

        # STATE 1: WAITING_FOR_RELEASE
        if self.state == BuilderState.WAITING_FOR_RELEASE:
            self.has_released_pose = False

            if not is_hand_present:
                if self.hand_disappeared_start_time is None:
                    self.hand_disappeared_start_time = now
                    self.status_message = "Hand absent. Waiting for release confirmation..."
                    return None
                else:
                    absent_elapsed = now - self.hand_disappeared_start_time
                    if absent_elapsed >= self.hand_release_duration_sec:
                        self.state = BuilderState.READY
                        self.has_released_pose = True
                        self.accepted_sign = None
                        self.accepted_landmarks = None
                        self.hand_disappeared_start_time = None
                        self.consecutive_sign = None
                        self.consecutive_count = 0
                        self.status_message = "Hand released. Ready for next sign."
                        return None
                    else:
                        self.status_message = f"Releasing hand ({int(absent_elapsed * 1000)}/{int(self.hand_release_duration_sec * 1000)}ms)..."
                        return None

            self.hand_disappeared_start_time = None

            has_gesture_changed = False
            if norm_landmarks is not None and self.accepted_landmarks is not None:
                pose_dist = self.compute_pose_distance(norm_landmarks, self.accepted_landmarks)
                if pose_dist >= self.landmark_diff_threshold:
                    has_gesture_changed = True
            elif predicted_sign != self.accepted_sign and is_valid_sign:
                has_gesture_changed = True

            if has_gesture_changed:
                self.state = BuilderState.READY
                self.has_released_pose = True
                self.accepted_sign = None
                self.accepted_landmarks = None
                self.consecutive_sign = None
                self.consecutive_count = 0
            else:
                self.status_message = f"Holding '{self.accepted_sign}'. Release hand or change sign to continue."
                return None

        # STATE 2: READY
        if self.state == BuilderState.READY:
            self.hand_disappeared_start_time = None

            if not is_hand_present:
                self.consecutive_sign = None
                self.consecutive_count = 0
                self.status_message = status_msg or "Show hand(s) to camera..."
                return None

            if not is_valid_sign:
                self.consecutive_sign = None
                self.consecutive_count = 0
                self.status_message = status_msg or "Position hand clearly in frame..."
                return None

            self.state = BuilderState.STABILIZING
            self.consecutive_sign = predicted_sign
            self.consecutive_count = 1
            self.status_message = f"Stabilizing '{predicted_sign}' (1/{self.stable_frames_required})..."
            return None

        # STATE 3: STABILIZING
        if self.state == BuilderState.STABILIZING:
            if not is_hand_present:
                self.consecutive_sign = None
                self.consecutive_count = 0
                self.state = BuilderState.READY
                self.status_message = status_msg or "Show hand(s) to camera..."
                return None

            if not is_valid_sign:
                self.consecutive_sign = None
                self.consecutive_count = 0
                self.state = BuilderState.READY
                self.status_message = status_msg or "Sign uncertain - hold steady..."
                return None

            if predicted_sign != self.consecutive_sign:
                self.consecutive_sign = predicted_sign
                self.consecutive_count = 1
                self.status_message = f"Stabilizing '{predicted_sign}' (1/{self.stable_frames_required})..."
                return None

            self.consecutive_count += 1
            self.status_message = f"Stabilizing '{predicted_sign}' ({self.consecutive_count}/{self.stable_frames_required})..."

            if self.consecutive_count >= self.stable_frames_required:
                self.state = BuilderState.ACCEPTED
            else:
                return None

        # STATE 4: ACCEPTED
        if self.state == BuilderState.ACCEPTED:
            sign_to_accept = self.consecutive_sign or predicted_sign
            if not sign_to_accept:
                self.state = BuilderState.READY
                return None

            new_char = self._append_character_internal(sign_to_accept, source="vision")

            self.accepted_sign = sign_to_accept
            if norm_landmarks is not None:
                self.accepted_landmarks = [np.copy(h) for h in norm_landmarks]
            else:
                self.accepted_landmarks = None

            self.accepted_timestamp = now
            self.last_added_sign = sign_to_accept
            self.last_addition_time = now
            self.has_released_pose = False

            self.consecutive_sign = None
            self.consecutive_count = 0
            self.hand_disappeared_start_time = None

            self.state = BuilderState.WAITING_FOR_RELEASE
            self.status_message = f"Accepted '{new_char}'. Release hand or change sign."
            return new_char

        return None

    def _append_character_internal(self, char: str, source: str = "manual") -> str:
        """Internal helper to append character, record history, and push undo."""
        self._push_undo_snapshot()
        char_upper = char.upper()
        self.current_word += char_upper

        source_desc = "Camera ISL" if source == "vision" else "Input"
        self._add_history_entry(
            "char",
            char_upper,
            f"Added '{char_upper}' ({source_desc})"
        )
        return char_upper

    def add_character(self, char: str) -> str:
        """Explicitly add a character."""
        char = str(char).strip().upper()
        if not char:
            return ""
        self.consecutive_sign = None
        self.consecutive_count = 0
        self.state = BuilderState.READY
        self.accepted_sign = None
        self.accepted_landmarks = None
        self.has_released_pose = True
        return self._append_character_internal(char[0], source="manual")

    def complete_word(self) -> str:
        """Confirms current word and appends it to completed_sentence with a space."""
        word = self.current_word.strip()
        if not word:
            return ""

        self._push_undo_snapshot()
        if self.completed_sentence and not self.completed_sentence.endswith(" "):
            self.completed_sentence += " "
        self.completed_sentence += word + " "

        self._add_history_entry("word", word, f"Completed word: {word}")
        self.current_word = ""
        self.last_added_sign = None
        self.consecutive_sign = None
        self.consecutive_count = 0
        self.state = BuilderState.READY
        self.accepted_sign = None
        self.accepted_landmarks = None
        self.has_released_pose = True

        self.word_suggestions = self.suggestion_engine.get_suggestions(self.completed_sentence)
        return word

    def delete_last_character(self) -> bool:
        """Removes the last character from current word, or last word from sentence."""
        self._push_undo_snapshot()
        self.consecutive_sign = None
        self.consecutive_count = 0
        self.state = BuilderState.READY
        self.accepted_sign = None
        self.accepted_landmarks = None
        self.has_released_pose = True

        if len(self.current_word) > 0:
            removed = self.current_word[-1]
            self.current_word = self.current_word[:-1]
            self._add_history_entry("edit", "DEL", f"Deleted '{removed}' from word")
            self.last_added_sign = None
            return True

        if len(self.completed_sentence) > 0:
            self.completed_sentence = self.completed_sentence.rstrip()
            if self.completed_sentence:
                words = self.completed_sentence.split()
                if words:
                    last_w = words.pop()
                    self.completed_sentence = (" ".join(words) + " ") if words else ""
                    self.current_word = last_w
                    self._add_history_entry("edit", "PULL", f"Pulled '{last_w}' back to editing")
                    self.word_suggestions = self.suggestion_engine.get_suggestions(self.completed_sentence)
                    return True
            return True

        return False

    def undo(self) -> bool:
        """Reverses the last state change using the undo stack."""
        if not self.undo_stack:
            return False

        last_state = self.undo_stack.pop()
        self.current_word = last_state["current_word"]
        self.completed_sentence = last_state["completed_sentence"]
        self.last_added_sign = last_state.get("last_added_sign")
        self.consecutive_sign = None
        self.consecutive_count = 0
        self.state = BuilderState.READY
        self.accepted_sign = None
        self.accepted_landmarks = None
        self.has_released_pose = True

        self._add_history_entry("edit", "UNDO", "Reversed last action")
        self.word_suggestions = self.suggestion_engine.get_suggestions(self.completed_sentence)
        return True

    def clear_current_word(self) -> bool:
        """Clears only the currently unconfirmed word."""
        if not self.current_word:
            return False
        self._push_undo_snapshot()
        cleared = self.current_word
        self.current_word = ""
        self.last_added_sign = None
        self.consecutive_sign = None
        self.consecutive_count = 0
        self.state = BuilderState.READY
        self.accepted_sign = None
        self.accepted_landmarks = None
        self.has_released_pose = True
        self._add_history_entry("edit", "CLEAR_WORD", f"Cleared word '{cleared}'")
        return True

    def clear_sentence(self) -> bool:
        """Clears both current word and completed sentence."""
        if not self.current_word and not self.completed_sentence:
            return False
        self._push_undo_snapshot()
        self.current_word = ""
        self.completed_sentence = ""
        self.last_added_sign = None
        self.consecutive_sign = None
        self.consecutive_count = 0
        self.state = BuilderState.READY
        self.accepted_sign = None
        self.accepted_landmarks = None
        self.has_released_pose = True
        self.word_suggestions = self.suggestion_engine.get_suggestions("")
        self._add_history_entry("edit", "CLEAR_ALL", "Cleared entire sentence")
        return True

    def apply_suggestion(self, word: str) -> str:
        """Applies a selected suggestion chip to the sentence."""
        clean_word = word.strip().upper()
        if not clean_word:
            return ""

        self._push_undo_snapshot()
        if self.completed_sentence and not self.completed_sentence.endswith(" "):
            self.completed_sentence += " "

        self.completed_sentence += clean_word + " "
        self.current_word = ""
        self.last_added_sign = None
        self.consecutive_sign = None
        self.consecutive_count = 0
        self.state = BuilderState.READY
        self.accepted_sign = None
        self.accepted_landmarks = None
        self.has_released_pose = True

        self._add_history_entry("suggestion", clean_word, f"Selected suggestion: {clean_word}")
        self.word_suggestions = self.suggestion_engine.get_suggestions(self.completed_sentence)
        return clean_word

    def remove_history_item(self, item_id: int) -> bool:
        """Remove a specific entry from recognition history."""
        initial_len = len(self.recognition_history)
        self.recognition_history = [item for item in self.recognition_history if item["id"] != item_id]
        return len(self.recognition_history) < initial_len

    def get_full_text(self) -> str:
        """Get the full displayed text (sentence + current word)."""
        base = self.completed_sentence
        if base and not base.endswith(" ") and self.current_word:
            base += " "
        return base + self.current_word

    def get_state(self) -> Dict[str, Any]:
        """Serialize complete application state for UI sync."""
        return {
            "current_word": self.current_word,
            "completed_sentence": self.completed_sentence,
            "full_sentence": self.get_full_text(),
            "word_suggestions": self.word_suggestions,
            "live_sign": self.live_sign,
            "live_confidence": round(self.live_confidence * 100, 1),
            "is_hands_detected": self.is_hands_detected,
            "hand_count": self.hand_count,
            "status_message": self.status_message,
            "builder_state": self.state,
            "stable_progress": f"{self.consecutive_count}/{self.stable_frames_required}" if self.state == BuilderState.STABILIZING else "",
            "recognition_history": self.recognition_history[:15],
            "can_undo": len(self.undo_stack) > 0,
            "has_content": bool(self.current_word or self.completed_sentence),
        }
