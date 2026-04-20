"""
Main Application Window
========================
Professional dark-themed medical image viewer.
Integrates 2D and 3D visualization for medical volumes.
"""

import os
import tempfile
import zipfile
import shutil
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QAction, QToolBar, QStatusBar, QLabel,
    QGroupBox, QPushButton,
    QFileDialog, QMessageBox, QTabWidget,
    QApplication
)
from PyQt5.QtCore import Qt, QTimer

from gui.viewer_2d import Viewer2DWidget
from gui.viewer_3d import Viewer3DWidget
from medical.loader import ImageLoader
from medical.preprocessing import generate_synthetic_volume
from utils.config import APP_NAME, APP_VERSION


class MainWindow(QMainWindow):
    """Main application window for a display-only medical viewer."""

    def __init__(self):
        super().__init__()
        self.medical_image = None

        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(900, 600)
        self.resize(1300, 700)

        self._setup_menu_bar()
        self._setup_toolbar()
        self._setup_central_widget()
        self._setup_status_bar()

        self.statusBar().showMessage("Ready - load a medical image or open the demo volume")

    # ──────────────────────────────────
    # MENU BAR
    # ──────────────────────────────────
    def _setup_menu_bar(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        load_file_action = QAction("📄 Load File (NIfTI/NRRD/MHD)...", self)
        load_file_action.setShortcut("Ctrl+O")
        load_file_action.triggered.connect(self._load_image)
        file_menu.addAction(load_file_action)

        load_folder_action = QAction("📂 Load DICOM Folder...", self)
        load_folder_action.setShortcut("Ctrl+Shift+O")
        load_folder_action.triggered.connect(self._load_dicom_folder)
        file_menu.addAction(load_folder_action)

        load_zip_action = QAction("📦 Load ZIP Archive...", self)
        load_zip_action.setShortcut("Ctrl+Z")
        load_zip_action.triggered.connect(self._load_zip)
        file_menu.addAction(load_zip_action)

        demo_action = QAction("🧪 Load Demo Volume", self)
        demo_action.setShortcut("Ctrl+D")
        demo_action.triggered.connect(self._load_demo)
        file_menu.addAction(demo_action)

        file_menu.addSeparator()

        quit_action = QAction("Exit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        vol3d_action = QAction("🔲 3D Volume Render", self)
        vol3d_action.setShortcut("F5")
        vol3d_action.triggered.connect(self._show_3d_volume)
        view_menu.addAction(vol3d_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    # ──────────────────────────────────
    # TOOLBAR
    # ──────────────────────────────────
    def _setup_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.addToolBar(toolbar)

        self.btn_load = QPushButton("📄 Load File")
        self.btn_load.clicked.connect(self._load_image)
        toolbar.addWidget(self.btn_load)

        self.btn_load_folder = QPushButton("📂 DICOM Folder")
        self.btn_load_folder.clicked.connect(self._load_dicom_folder)
        toolbar.addWidget(self.btn_load_folder)

        self.btn_load_zip = QPushButton("📦 Load ZIP")
        self.btn_load_zip.clicked.connect(self._load_zip)
        toolbar.addWidget(self.btn_load_zip)

        self.btn_demo = QPushButton("🧪 Demo")
        self.btn_demo.clicked.connect(self._load_demo)
        toolbar.addWidget(self.btn_demo)

        toolbar.addSeparator()

        self.btn_3d_vol = QPushButton("🔲 3D Volume")
        self.btn_3d_vol.clicked.connect(self._show_3d_volume)
        self.btn_3d_vol.setEnabled(False)
        toolbar.addWidget(self.btn_3d_vol)

    # ──────────────────────────────────
    # CENTRAL WIDGET
    # ──────────────────────────────────
    def _setup_central_widget(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # Main splitter: [Viewers | Side Panel]
        splitter = QSplitter(Qt.Horizontal)

        # Viewer tabs (2D + 3D)
        self.viewer_tabs = QTabWidget()
        self.viewer_2d = Viewer2DWidget()
        self.viewer_3d = Viewer3DWidget()
        self.viewer_tabs.addTab(self.viewer_2d, "📊 2D Slices")
        self.viewer_tabs.addTab(self.viewer_3d, "🔲 3D View")

        splitter.addWidget(self.viewer_tabs)

        # Side panel
        side_panel = self._create_side_panel()
        splitter.addWidget(side_panel)

        splitter.setSizes([1000, 350])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

    def _create_side_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(300)
        panel.setMaximumWidth(420)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)

        # ── Image Info ──
        info_group = QGroupBox("📋 Image Info")
        info_layout = QVBoxLayout(info_group)

        self.lbl_filename = QLabel("File: —")
        self.lbl_dimensions = QLabel("Dimensions: —")
        self.lbl_spacing = QLabel("Spacing: —")
        self.lbl_dtype = QLabel("Type: —")

        for lbl in [self.lbl_filename, self.lbl_dimensions, self.lbl_spacing, self.lbl_dtype]:
            lbl.setStyleSheet("font-size: 11px;")
            info_layout.addWidget(lbl)

        layout.addWidget(info_group)

        # ── Viewer Guide ──
        guide_group = QGroupBox("👁 Viewer Guide")
        guide_layout = QVBoxLayout(guide_group)

        self.lbl_guide = QLabel(
            "Load DICOM, NIfTI, NRRD, or MHD data to explore slices in 2D and render the volume in 3D."
        )
        self.lbl_guide.setWordWrap(True)
        self.lbl_guide.setStyleSheet("font-size: 11px; line-height: 1.4;")
        guide_layout.addWidget(self.lbl_guide)

        layout.addWidget(guide_group)

        layout.addStretch()
        return panel

    # ──────────────────────────────────
    # STATUS BAR
    # ──────────────────────────────────
    def _setup_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.status_label = QLabel("")
        self.status_bar.addPermanentWidget(self.status_label)

    # ──────────────────────────────────
    # ACTIONS
    # ──────────────────────────────────
    def _load_image(self):
        """Open file dialog to load a single medical image file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Medical Image File", "",
            "Medical Images (*.nii *.nii.gz *.nrrd *.mhd *.dcm);;NIfTI (*.nii *.nii.gz);;NRRD (*.nrrd);;MHD (*.mhd);;DICOM (*.dcm);;All Files (*.*)"
        )
        if not path:
            return

        try:
            self.statusBar().showMessage(f"Loading {os.path.basename(path)}...")
            QApplication.processEvents()

            # If user selected a single .dcm file, load the parent folder as DICOM series
            if path.lower().endswith('.dcm'):
                load_dir = os.path.dirname(path)
                self.medical_image = ImageLoader.load_dicom_series(load_dir)
            else:
                self.medical_image = ImageLoader.load_auto(path)

            self._on_image_loaded(os.path.basename(path))

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load image:\n{str(e)}")
            self.statusBar().showMessage("Load failed")

    def _load_dicom_folder(self):
        """Open folder dialog to load a DICOM series from a directory."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select DICOM Folder", "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if not folder:
            return

        try:
            self.statusBar().showMessage(f"Loading DICOM series from {folder}...")
            QApplication.processEvents()

            self.medical_image = ImageLoader.load_dicom_series(folder)
            self._on_image_loaded(f"DICOM: {os.path.basename(folder)}")

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load DICOM folder:\n{str(e)}")
            self.statusBar().showMessage("Load failed")

    def _load_zip(self):
        """Load a ZIP archive containing medical images (DICOM folder or NIfTI files)."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load ZIP Archive", "",
            "ZIP Archives (*.zip);;All Files (*.*)"
        )
        if not path:
            return

        try:
            self.statusBar().showMessage(f"Extracting {os.path.basename(path)}...")
            QApplication.processEvents()

            # Extract to a temp directory
            temp_dir = tempfile.mkdtemp(prefix="medai_")
            with zipfile.ZipFile(path, 'r') as zf:
                zf.extractall(temp_dir)

            self.statusBar().showMessage("Scanning extracted files...")
            QApplication.processEvents()

            # Try to find medical data in the extracted contents
            loaded = False

            # 1) Look for NIfTI / NRRD / MHD files first
            for root, dirs, files in os.walk(temp_dir):
                for f in files:
                    fl = f.lower()
                    if fl.endswith(('.nii', '.nii.gz', '.nrrd', '.mhd')):
                        fpath = os.path.join(root, f)
                        self.medical_image = ImageLoader.load_auto(fpath)
                        self._on_image_loaded(f"ZIP: {f}")
                        loaded = True
                        break
                if loaded:
                    break

            # 2) If no NIfTI found, try to find DICOM series
            if not loaded:
                for root, dirs, files in os.walk(temp_dir):
                    dcm_files = [f for f in files if f.lower().endswith('.dcm')
                                 or not os.path.splitext(f)[1]]  # DICOM files often have no extension
                    if len(dcm_files) > 5:  # Likely a DICOM series
                        try:
                            self.medical_image = ImageLoader.load_dicom_series(root)
                            self._on_image_loaded(f"ZIP DICOM: {os.path.basename(root)}")
                            loaded = True
                            break
                        except Exception:
                            continue

            if not loaded:
                QMessageBox.warning(
                    self, "No Medical Data",
                    "No supported medical images found in the ZIP archive.\n\n"
                    "Supported formats: DICOM series, NIfTI (.nii/.nii.gz), NRRD, MHD"
                )
                self.statusBar().showMessage("No medical data found in ZIP")
                # Clean up temp dir
                shutil.rmtree(temp_dir, ignore_errors=True)

        except zipfile.BadZipFile:
            QMessageBox.critical(self, "Load Error", "The selected file is not a valid ZIP archive.")
            self.statusBar().showMessage("Invalid ZIP file")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load ZIP:\n{str(e)}")
            self.statusBar().showMessage("ZIP load failed")

    def _load_demo(self):
        """Load a synthetic demo volume."""
        self.statusBar().showMessage("Generating synthetic demo volume...")
        QApplication.processEvents()

        self.medical_image, self._demo_truth = generate_synthetic_volume(
            size=(128, 128, 64), spacing=(1.0, 1.0, 2.0)
        )
        self._on_image_loaded("Synthetic Demo (128×128×64)")

    def _on_image_loaded(self, name: str):
        """Called after a medical image is successfully loaded."""
        img = self.medical_image

        # Update info panel
        self.lbl_filename.setText(f"File: {name}")
        self.lbl_dimensions.setText(
            f"Dimensions: {img.size[0]} × {img.size[1]} × {img.size[2]}"
        )
        self.lbl_spacing.setText(
            f"Spacing: {img.spacing[0]:.2f} × {img.spacing[1]:.2f} × {img.spacing[2]:.2f} mm"
        )
        self.lbl_dtype.setText(f"Type: {img.volume.dtype}")

        # Update 2D viewer
        self.viewer_2d.set_volume(img.volume)

        # Update 3D viewer
        self.viewer_3d.set_volume(img.volume, tuple(img.spacing))

        # Enable buttons
        self.btn_3d_vol.setEnabled(True)

        self.statusBar().showMessage(
            f"Loaded: {name} | Shape: {img.shape} | Spacing: {img.spacing.tolist()}"
        )

    def _show_3d_volume(self):
        """Switch to 3D tab and show volume rendering."""
        self.viewer_tabs.setCurrentIndex(1)
        self.viewer_3d._show_volume_rendering()

    def _show_about(self):
        QMessageBox.about(
            self, f"About {APP_NAME}",
            f"<h2>{APP_NAME}</h2>"
            f"<p>Version {APP_VERSION}</p>"
            f"<p>Desktop application for loading and viewing medical image volumes.</p>"
            f"<hr>"
            f"<p><b>Features:</b></p>"
            f"<ul>"
            f"<li>DICOM / NIfTI / NRRD / MHD support</li>"
            f"<li>Three-plane 2D viewer (Axial/Sagittal/Coronal)</li>"
            f"<li>VTK 3D Volume Rendering</li>"
            f"<li>Window/level, brightness, contrast, and inversion controls</li>"
            f"<li>Built-in synthetic demo volume</li>"
            f"</ul>"
            f"<p><b>Built with:</b> PyQt5, VTK, SimpleITK</p>"
        )

    def showEvent(self, event):
        """Initialize VTK after the window is shown."""
        super().showEvent(event)
        QTimer.singleShot(100, self.viewer_3d.initialize)
