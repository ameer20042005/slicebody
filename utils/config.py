"""
Application Configuration
=========================
Colors, presets, and settings for the Medical AI Desktop App.
"""

# ──────────────────────────────────────
# Dark Theme Stylesheet (Medical UI)
# ──────────────────────────────────────
DARK_STYLESHEET = """
QMainWindow {
    background-color: #1a1a2e;
}

QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: "Segoe UI", "Roboto", "Arial", sans-serif;
    font-size: 13px;
}

QMenuBar {
    background-color: #16213e;
    color: #e0e0e0;
    border-bottom: 1px solid #0f3460;
    padding: 4px;
}

QMenuBar::item:selected {
    background-color: #0f3460;
    border-radius: 4px;
}

QMenu {
    background-color: #16213e;
    color: #e0e0e0;
    border: 1px solid #0f3460;
}

QMenu::item:selected {
    background-color: #0f3460;
}

QToolBar {
    background-color: #16213e;
    border: none;
    padding: 4px;
    spacing: 6px;
}

QToolButton {
    background-color: #0f3460;
    color: #e0e0e0;
    border: 1px solid #1a1a5e;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: bold;
    min-width: 80px;
}

QToolButton:hover {
    background-color: #e94560;
    border-color: #e94560;
}

QToolButton:pressed {
    background-color: #c73e54;
}

QToolButton:disabled {
    background-color: #2a2a4a;
    color: #666;
}

QPushButton {
    background-color: #0f3460;
    color: #e0e0e0;
    border: 1px solid #1a1a5e;
    border-radius: 6px;
    padding: 8px 20px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #e94560;
    border-color: #e94560;
}

QPushButton:pressed {
    background-color: #c73e54;
}

QPushButton#btnSegment {
    background-color: #00b894;
    border-color: #00a884;
    font-size: 14px;
    padding: 10px 24px;
}

QPushButton#btnSegment:hover {
    background-color: #00d4aa;
}

QLabel {
    color: #e0e0e0;
    background: transparent;
}

QLabel#titleLabel {
    font-size: 18px;
    font-weight: bold;
    color: #00cec9;
}

QLabel#sectionLabel {
    font-size: 14px;
    font-weight: bold;
    color: #74b9ff;
    border-bottom: 1px solid #0f3460;
    padding-bottom: 4px;
    margin-top: 8px;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #2d2d5e;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #e94560;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}

QSlider::sub-page:horizontal {
    background: #0f3460;
    border-radius: 3px;
}

QComboBox {
    background-color: #16213e;
    color: #e0e0e0;
    border: 1px solid #0f3460;
    border-radius: 4px;
    padding: 6px 10px;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    background-color: #16213e;
    color: #e0e0e0;
    selection-background-color: #0f3460;
}

QSpinBox, QDoubleSpinBox {
    background-color: #16213e;
    color: #e0e0e0;
    border: 1px solid #0f3460;
    border-radius: 4px;
    padding: 4px 8px;
}

QGroupBox {
    border: 1px solid #0f3460;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 20px;
    font-weight: bold;
    color: #74b9ff;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}

QStatusBar {
    background-color: #0d1b2a;
    color: #74b9ff;
    border-top: 1px solid #0f3460;
    font-size: 12px;
}

QProgressBar {
    background-color: #2d2d5e;
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: #00b894;
    border-radius: 4px;
}

QSplitter::handle {
    background-color: #0f3460;
    width: 3px;
}

QTextEdit {
    background-color: #0d1b2a;
    color: #00cec9;
    border: 1px solid #0f3460;
    border-radius: 6px;
    padding: 8px;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
}

QScrollBar:vertical {
    background: #1a1a2e;
    width: 10px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background: #0f3460;
    border-radius: 5px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background: #e94560;
}

QTabWidget::pane {
    border: 1px solid #0f3460;
    border-radius: 6px;
    background-color: #1a1a2e;
}

QTabBar::tab {
    background-color: #16213e;
    color: #a0a0c0;
    border: 1px solid #0f3460;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 8px 16px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #0f3460;
    color: #e0e0e0;
    font-weight: bold;
}

QTabBar::tab:hover {
    background-color: #1a2a5e;
}
"""

# ──────────────────────────────────────
# Segmentation Mask Colors (R, G, B, Alpha)
# ──────────────────────────────────────
MASK_COLORS = {
    "default":  (230, 69, 96, 100),    # Red-pink
    "tumor":    (255, 99, 71, 120),    # Tomato red
    "organ":    (0, 184, 148, 100),    # Green
    "bone":     (255, 234, 167, 100),  # Yellow
    "vessel":   (116, 185, 255, 100),  # Blue
}

# ──────────────────────────────────────
# VTK Volume Rendering Presets
# ──────────────────────────────────────
VTK_PRESETS = {
    "CT Bone": {
        "opacity": [(-1000, 0.0), (-100, 0.0), (200, 0.1), (500, 0.6), (1000, 0.9)],
        "color": [(-1000, (0, 0, 0)), (200, (0.9, 0.8, 0.6)), (500, (1, 1, 0.9)), (1000, (1, 1, 1))],
    },
    "CT Soft Tissue": {
        "opacity": [(-1000, 0.0), (-200, 0.0), (0, 0.05), (100, 0.4), (300, 0.7)],
        "color": [(-1000, (0, 0, 0)), (0, (0.8, 0.3, 0.3)), (100, (0.9, 0.6, 0.5)), (300, (1, 0.9, 0.8))],
    },
    "MRI Default": {
        "opacity": [(0, 0.0), (200, 0.0), (500, 0.2), (800, 0.5), (1200, 0.8)],
        "color": [(0, (0, 0, 0)), (500, (0.5, 0.5, 0.8)), (800, (0.8, 0.7, 0.9)), (1200, (1, 1, 1))],
    },
}

# ──────────────────────────────────────
# Application Info
# ──────────────────────────────────────
APP_NAME = "Medical AI Viewer"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Medical Image Analysis & AI Segmentation Desktop Application"
