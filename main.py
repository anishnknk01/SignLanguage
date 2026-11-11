#!/usr/bin/env python3
"""
Entrypoint for the ISL Sign Language Recognition application.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

if __name__ == "__main__":
    if "--cv2" in sys.argv:
        import runpy
        runpy.run_path(os.path.join(SRC_DIR, "detect_cnn.py"), run_name="__main__")
    else:
        from gui_local import main
        main()
