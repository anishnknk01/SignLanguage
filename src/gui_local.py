#!/usr/bin/env python3
"""
ISL Sign-to-Text and Sentence Formation Desktop Application.
"""

import os
import sys
import time
import subprocess
import threading
from typing import Optional
import cv2
import numpy as np
from PIL import Image
import customtkinter as ctk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from recognition_engine import ISLRecognitionEngine
from sentence_builder import SentenceBuilder

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ISLDesktopApp(ctk.CTk):
    """Desktop GUI for Indian Sign Language recognition and sentence building."""

    def __init__(self):
        super().__init__()

        self.title("ISL Sign Language Recognition")
        self.geometry("1340x820")
        self.minsize(1080, 680)
        self.configure(fg_color="#0A0E17")

        # Core Engines
        self.recognition_engine = ISLRecognitionEngine(confidence_threshold=0.70)
        self.sentence_builder = SentenceBuilder(
            stable_frames_required=10,
            hand_release_duration_sec=0.25,
            landmark_diff_threshold=0.10,
            min_confidence=0.70
        )

        # Camera & Processing State
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_camera_running = False
        self.camera_thread: Optional[threading.Thread] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        self.is_closing = False
        self.last_toast_time = 0

        # Build UI Components
        self._setup_ui()
        self._setup_keyboard_shortcuts()

        # Start Camera Stream & Frame Polling
        self._start_camera()
        self._poll_vision_and_ui()

    def _setup_ui(self):
        """Constructs the 2-column layout."""
        # Top Header
        self.header_frame = ctk.CTkFrame(
            self, height=56, corner_radius=0, fg_color="#121824",
            border_width=1, border_color="#1E293B"
        )
        self.header_frame.pack(side="top", fill="x", padx=0, pady=0)

        brand_box = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        brand_box.pack(side="left", padx=20, pady=10)

        title_lbl = ctk.CTkLabel(
            brand_box,
            text="ISL Sign Recognition",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#F8FAFC"
        )
        title_lbl.pack(side="left", padx=(0, 12))

        sub_lbl = ctk.CTkLabel(
            brand_box,
            text="Real-Time Sentence Builder",
            font=ctk.CTkFont(size=12),
            text_color="#64748B"
        )
        sub_lbl.pack(side="left")

        # Header Right: Camera Status & Toast
        status_box = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        status_box.pack(side="right", padx=20, pady=10)

        self.toast_label = ctk.CTkLabel(
            status_box,
            text="",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#00D2FF"
        )
        self.toast_label.pack(side="left", padx=(0, 16))

        self.cam_status_lbl = ctk.CTkLabel(
            status_box,
            text="Camera Active",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#10B981"
        )
        self.cam_status_lbl.pack(side="left")

        # Main 2-Column Container
        self.main_container = ctk.CTkFrame(self, fg_color="#0A0E17", corner_radius=0)
        self.main_container.pack(fill="both", expand=True, padx=16, pady=14)
        self.main_container.grid_columnconfigure(0, weight=6)
        self.main_container.grid_columnconfigure(1, weight=5)
        self.main_container.grid_rowconfigure(0, weight=1)

        # Left Column: Live Sign Detection
        self.left_col = ctk.CTkFrame(
            self.main_container, fg_color="#121824", corner_radius=12,
            border_width=1, border_color="#1E293B"
        )
        self.left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=0)

        cam_hdr = ctk.CTkFrame(self.left_col, fg_color="transparent")
        cam_hdr.pack(fill="x", padx=16, pady=(12, 6))
        ctk.CTkLabel(
            cam_hdr,
            text="Camera Feed",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#E2E8F0"
        ).pack(side="left")

        self.video_display = ctk.CTkLabel(
            self.left_col,
            text="Initializing Camera Stream...",
            fg_color="#080C14",
            corner_radius=8,
            width=680,
            height=510
        )
        self.video_display.pack(padx=16, pady=8, fill="both", expand=True)

        # Detection Card below video
        det_bar = ctk.CTkFrame(
            self.left_col, fg_color="#0D111A", corner_radius=8,
            border_width=1, border_color="#1E293B", height=64
        )
        det_bar.pack(fill="x", padx=16, pady=(6, 14))

        self.live_sign_lbl = ctk.CTkLabel(
            det_bar,
            text="-",
            font=ctk.CTkFont(family="Courier", size=32, weight="bold"),
            text_color="#00D2FF",
            width=54
        )
        self.live_sign_lbl.pack(side="left", padx=(12, 6), pady=8)

        sign_info = ctk.CTkFrame(det_bar, fg_color="transparent")
        sign_info.pack(side="left", fill="both", expand=True, padx=4, pady=8)

        ctk.CTkLabel(
            sign_info,
            text="PREDICTED SIGN",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color="#64748B",
            anchor="w"
        ).pack(anchor="w")

        self.live_conf_lbl = ctk.CTkLabel(
            sign_info,
            text="Confidence: -",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#94A3B8",
            anchor="w"
        )
        self.live_conf_lbl.pack(anchor="w")

        self.detection_msg_lbl = ctk.CTkLabel(
            det_bar,
            text="Position hand in camera view",
            font=ctk.CTkFont(size=11),
            text_color="#64748B"
        )
        self.detection_msg_lbl.pack(side="right", padx=14)

        # Right Column: Sentence Builder
        self.right_col = ctk.CTkFrame(
            self.main_container, fg_color="#121824", corner_radius=12,
            border_width=1, border_color="#1E293B"
        )
        self.right_col.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=0)

        sb_hdr = ctk.CTkFrame(self.right_col, fg_color="transparent")
        sb_hdr.pack(fill="x", padx=16, pady=(12, 6))
        ctk.CTkLabel(
            sb_hdr,
            text="Sentence Builder",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#E2E8F0"
        ).pack(side="left")

        # 1. Current Word Box
        word_box = ctk.CTkFrame(
            self.right_col, fg_color="#0D111A", corner_radius=8,
            border_width=1, border_color="#1E293B"
        )
        word_box.pack(fill="x", padx=16, pady=6)

        wb_hdr = ctk.CTkFrame(word_box, fg_color="transparent")
        wb_hdr.pack(fill="x", padx=12, pady=(8, 2))
        ctk.CTkLabel(
            wb_hdr,
            text="CURRENT WORD",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#64748B"
        ).pack(side="left")

        self.char_count_lbl = ctk.CTkLabel(
            wb_hdr,
            text="0 chars",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#00D2FF"
        )
        self.char_count_lbl.pack(side="right")

        self.current_word_display = ctk.CTkLabel(
            word_box,
            text="-",
            font=ctk.CTkFont(family="Courier", size=32, weight="bold"),
            text_color="#00D2FF",
            anchor="w"
        )
        self.current_word_display.pack(fill="x", padx=12, pady=(2, 10))

        # 2. Completed Sentence Box
        sent_box = ctk.CTkFrame(
            self.right_col, fg_color="#080C14", corner_radius=8,
            border_width=1, border_color="#1E293B"
        )
        sent_box.pack(fill="x", padx=16, pady=6)

        sb_actions = ctk.CTkFrame(sent_box, fg_color="transparent")
        sb_actions.pack(fill="x", padx=12, pady=(8, 2))
        ctk.CTkLabel(
            sb_actions,
            text="COMPLETED SENTENCE",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#64748B"
        ).pack(side="left")

        copy_btn = ctk.CTkButton(
            sb_actions,
            text="Copy",
            width=62,
            height=24,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#1E293B",
            hover_color="#334155",
            text_color="#E2E8F0",
            corner_radius=6,
            command=self._copy_sentence
        )
        copy_btn.pack(side="right")

        self.sentence_display = ctk.CTkLabel(
            sent_box,
            text="Completed words will appear here.",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#F8FAFC",
            wraplength=480,
            justify="left",
            anchor="w"
        )
        self.sentence_display.pack(fill="x", padx=12, pady=(4, 14))

        # 3. Suggested Next Words Section
        sug_box = ctk.CTkFrame(
            self.right_col, fg_color="#0D111A", corner_radius=8,
            border_width=1, border_color="#1E293B"
        )
        sug_box.pack(fill="x", padx=16, pady=6)

        sug_hdr = ctk.CTkFrame(sug_box, fg_color="transparent")
        sug_hdr.pack(fill="x", padx=12, pady=(8, 4))
        ctk.CTkLabel(
            sug_hdr,
            text="SUGGESTED NEXT WORDS (Click or 1-5):",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#64748B"
        ).pack(side="left")

        self.suggestion_buttons_frame = ctk.CTkFrame(sug_box, fg_color="transparent")
        self.suggestion_buttons_frame.pack(fill="x", padx=8, pady=(2, 8))

        self.suggestion_buttons = []
        for i in range(5):
            btn = ctk.CTkButton(
                self.suggestion_buttons_frame,
                text="",
                width=88,
                height=32,
                font=ctk.CTkFont(size=11, weight="bold"),
                fg_color="#1E293B",
                hover_color="#0284C7",
                corner_radius=6,
                command=lambda idx=i: self._on_suggestion_clicked(idx)
            )
            btn.pack(side="left", padx=3, expand=True)
            self.suggestion_buttons.append(btn)

        # 4. Primary Controls: Complete Word, Delete, Clear
        ctrl_frame = ctk.CTkFrame(self.right_col, fg_color="transparent")
        ctrl_frame.pack(fill="x", padx=16, pady=(12, 14))

        btn_complete = ctk.CTkButton(
            ctrl_frame,
            text="Complete Word",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#0284C7",
            hover_color="#00D2FF",
            text_color="#FFFFFF",
            corner_radius=8,
            height=42,
            command=self._on_complete_word
        )
        btn_complete.pack(side="left", fill="x", expand=True, padx=(0, 4))

        btn_del = ctk.CTkButton(
            ctrl_frame,
            text="Delete",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#1E293B",
            hover_color="#334155",
            text_color="#E2E8F0",
            corner_radius=8,
            height=42,
            command=self._on_delete
        )
        btn_del.pack(side="left", fill="x", expand=True, padx=4)

        btn_clear = ctk.CTkButton(
            ctrl_frame,
            text="Clear",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#2A1B24",
            hover_color="#4C1D24",
            text_color="#FCA5A5",
            corner_radius=8,
            height=42,
            command=self._on_clear
        )
        btn_clear.pack(side="left", fill="x", expand=True, padx=(4, 0))

    def _setup_keyboard_shortcuts(self):
        """Binds desktop keyboard shortcuts."""
        self.bind("<space>", lambda e: self._on_complete_word())
        self.bind("<Return>", lambda e: self._on_complete_word())
        self.bind("<BackSpace>", lambda e: self._on_delete())
        self.bind("<Escape>", lambda e: self._on_clear())

        for k in range(1, 6):
            self.bind(str(k), lambda e, idx=k-1: self._on_suggestion_clicked(idx))

    def _start_camera(self):
        """Starts background webcam reader."""
        self.is_camera_running = True
        self.camera_thread = threading.Thread(target=self._camera_worker, daemon=True)
        self.camera_thread.start()

    def _camera_worker(self):
        """Continuously pulls frames from cv2.VideoCapture(0)."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self._update_cam_status(False)
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._update_cam_status(True)

        while self.is_camera_running and not self.is_closing:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.04)
                continue
            with self.frame_lock:
                self.latest_frame = frame
            time.sleep(0.015)

        if self.cap:
            self.cap.release()

    def _update_cam_status(self, is_online: bool):
        if is_online:
            self.cam_status_lbl.configure(text="Camera Active", text_color="#10B981")
        else:
            self.cam_status_lbl.configure(text="Camera Offline", text_color="#EF4444")

    def _poll_vision_and_ui(self):
        """Main periodic loop: runs CNN inference and synchronizes UI."""
        if self.is_closing:
            return

        frame = None
        with self.frame_lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()

        if frame is not None:
            res = self.recognition_engine.process_frame(frame, flip=True)
            out_frame, pred_sign, conf, num_hands, status_msg = res[:5]
            landmarks = getattr(res, "landmarks", self.recognition_engine.latest_landmarks)

            self.sentence_builder.update_live_detection(
                predicted_sign=pred_sign,
                confidence=conf,
                num_hands=num_hands,
                status_msg=status_msg,
                landmarks=landmarks
            )

            h, w, _ = out_frame.shape
            target_w = 680
            target_h = int(h * (target_w / w))
            resized = cv2.resize(out_frame, (target_w, target_h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(target_w, target_h))

            self.video_display.ctk_img = ctk_img
            self.video_display.configure(image=ctk_img, text="")

        state = self.sentence_builder.get_state()

        sign = state["live_sign"]
        conf_pct = state["live_confidence"]
        self.live_sign_lbl.configure(text=sign)
        if conf_pct >= 70 and sign != "-":
            self.live_sign_lbl.configure(text_color="#00D2FF")
            self.live_conf_lbl.configure(text=f"Confidence: {conf_pct:.0f}%")
        elif sign != "-":
            self.live_sign_lbl.configure(text_color="#38BDF8")
            self.live_conf_lbl.configure(text=f"Confidence: {conf_pct:.0f}%")
        else:
            self.live_sign_lbl.configure(text_color="#64748B")
            self.live_conf_lbl.configure(text="Confidence: -")

        if not state["is_hands_detected"]:
            self.detection_msg_lbl.configure(text="Position hand in camera view", text_color="#64748B")
        else:
            self.detection_msg_lbl.configure(text=state["status_message"], text_color="#94A3B8")

        curr_w = state["current_word"]
        if curr_w:
            self.current_word_display.configure(text=curr_w + " |", text_color="#00D2FF")
            self.char_count_lbl.configure(text=f"{len(curr_w)} char{'s' if len(curr_w) > 1 else ''}")
        else:
            self.current_word_display.configure(text="-", text_color="#475569")
            self.char_count_lbl.configure(text="0 chars")

        sent_txt = state["completed_sentence"].strip()
        if sent_txt:
            self.sentence_display.configure(text=sent_txt, text_color="#F8FAFC")
        else:
            self.sentence_display.configure(
                text="Completed words will appear here.",
                text_color="#64748B"
            )

        suggs = state["word_suggestions"]
        for i in range(5):
            btn = self.suggestion_buttons[i]
            if i < len(suggs):
                w = suggs[i]
                btn.configure(text=f"{i+1}. {w}", state="normal", fg_color="#1E293B", text_color="#E2E8F0")
            else:
                btn.configure(text="", state="disabled", fg_color="#0D111A", text_color="#475569")

        self.after(25, self._poll_vision_and_ui)

    def _show_toast(self, msg: str):
        self.toast_label.configure(text=msg)
        self.last_toast_time = time.time()
        self.after(1800, lambda: self.toast_label.configure(text="") if time.time() - self.last_toast_time >= 1.7 else None)

    def _on_complete_word(self):
        """Confirms current word and advances sentence."""
        w = self.sentence_builder.complete_word()
        if w:
            self._show_toast(f"Completed '{w}'")

    def _on_delete(self):
        """Removes the most recent character or word."""
        if self.sentence_builder.delete_last_character():
            self._show_toast("Deleted")

    def _on_clear(self):
        """Clears current word if in progress; otherwise clears entire sentence."""
        if self.sentence_builder.current_word:
            self.sentence_builder.clear_current_word()
            self._show_toast("Word cleared")
        elif self.sentence_builder.completed_sentence:
            self.sentence_builder.clear_sentence()
            self._show_toast("Sentence cleared")

    def _on_suggestion_clicked(self, idx: int):
        """Applies clicked next-word suggestion."""
        suggs = self.sentence_builder.word_suggestions
        if 0 <= idx < len(suggs):
            word = suggs[idx]
            self.sentence_builder.apply_suggestion(word)
            self._show_toast(f"Added: {word}")

    def _copy_sentence(self):
        """Copies full sentence to clipboard."""
        text = self.sentence_builder.get_full_text().strip()
        if not text:
            self._show_toast("Sentence is empty")
            return

        try:
            p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            p.communicate(text.encode("utf-8"))
        except Exception:
            pass

        self.clipboard_clear()
        self.clipboard_append(text)
        self._show_toast("Copied to clipboard")

    def destroy(self):
        """Clean shutdown."""
        self.is_closing = True
        self.is_camera_running = False
        if self.cap:
            self.cap.release()
        super().destroy()


def main():
    print("Starting ISL Sign Recognition GUI...")
    app = ISLDesktopApp()
    app.mainloop()


if __name__ == "__main__":
    main()
