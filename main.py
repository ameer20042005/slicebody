"""
Medical AI Viewer — Entry Point
================================
Professional Medical Image Analysis & AI Segmentation Desktop Application.

Usage:
    python main.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from gui.main_window import MainWindow
from utils.config import DARK_STYLESHEET, APP_NAME


def main():
    # High DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)

    # Apply dark medical theme
    app.setStyleSheet(DARK_STYLESHEET)

    # Set default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Create and show main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()