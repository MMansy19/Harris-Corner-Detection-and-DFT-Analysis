import cv2
from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QFileDialog,
                             QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout,
                             QFormLayout, QTextEdit, QSplitter, QFrame,
                             QMessageBox, QCheckBox, QTabWidget, QSlider,
                             QRadioButton, QButtonGroup, QComboBox, QSpinBox,
                             QDoubleSpinBox, QStackedWidget)
from PyQt5.QtGui import QFont, QPixmap, QColor, QPalette, QIcon
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtCore import pyqtSignal

# Import the required functions from functions
from functions import (dft_analysis,
                     np_to_pixmap,
                     combine_magnitude_phase,
                     reconstruct_custom_image,
                     calculate_reconstruction_metrics,
                     harris_corner_detector)
class ClickableQLabel(QLabel):
    clicked = pyqtSignal()  # Signal emitted when clicked
    
    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

class ImageProcessorClass(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Analysis: DFT & Harris Corner Detection")
        self.resize(1400, 900)
        
        # Initialize image variables
        self.img1 = None
        self.img2 = None
        self.img1_dft_components = None
        self.img2_dft_components = None
        self.combined_img = None
        self.corner_img1=None
        self.corner_img2=None
        self.response_vis1=None
        self.response_vis2=None
        
        # Setup the UI
        self.setup_ui()
        self.apply_styles()
    
    def setup_ui(self):
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Create mode selector at the top
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(10)
        
        mode_label = QLabel("Analysis Mode:")
        mode_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["DFT Analysis", "Harris Corner Detection"])
        self.mode_combo.setMinimumWidth(200)
        self.mode_combo.setMinimumHeight(30)
        self.mode_combo.currentIndexChanged.connect(self.switch_mode)
        
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch(1)
        
        # Add mode selector to main layout
        main_layout.addLayout(mode_layout)
        
        # Create a horizontal split for left panel (images) and right panel (controls)
        main_splitter = QSplitter(Qt.Horizontal)
        
        # ========= LEFT PANEL (IMAGES) =========
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create top image grid (original images)
        top_group = QGroupBox("Input Images")
        top_group.setStyleSheet("QGroupBox {font-weight: bold; font-size: 14px;}")
        top_layout = QGridLayout()
        top_layout.setSpacing(10)
        
        # Image labels for original images
        self.label_img1 = QLabel("Load Image 1")
        self.label_img2 = QLabel("Load Image 2")
        
        # Set alignment and styling
        for label in [self.label_img1, self.label_img2]:
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(350, 300)
            label.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        
        # Add to grid
        top_layout.addWidget(QLabel("Image 1:"), 0, 0, Qt.AlignLeft)
        top_layout.addWidget(self.label_img1, 1, 0)
        top_layout.addWidget(QLabel("Image 2:"), 0, 1, Qt.AlignLeft)
        top_layout.addWidget(self.label_img2, 1, 1)
        
        top_group.setLayout(top_layout)
        
        # Create a stacked widget for bottom content based on mode
        self.bottom_stacked = QStackedWidget()
        
        # ---- DFT Mode Bottom Panel ----
        dft_bottom_widget = QWidget()
        dft_bottom_layout = QGridLayout(dft_bottom_widget)
        dft_bottom_layout.setSpacing(10)
        
        # Image labels for spectra
        self.label_mag1 = QLabel("Magnitude Spectrum 1")
        self.label_phase1 = QLabel("Phase Spectrum 1")
        self.label_mag2 = QLabel("Magnitude Spectrum 2")
        self.label_phase2 = QLabel("Phase Spectrum 2")
        self.label_dft_result = QLabel("Combined Result")
        
        # Set alignment and styling
        for label in [self.label_mag1, self.label_phase1, self.label_mag2, self.label_phase2, self.label_dft_result]:
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(275, 220)
            label.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        
        # Add to grid
        dft_bottom_layout.addWidget(QLabel("Mag 1:"), 0, 0, Qt.AlignLeft)
        dft_bottom_layout.addWidget(self.label_mag1, 1, 0)
        dft_bottom_layout.addWidget(QLabel("Phase 1:"), 0, 1, Qt.AlignLeft)
        dft_bottom_layout.addWidget(self.label_phase1, 1, 1)
        dft_bottom_layout.addWidget(QLabel("Mag 2:"), 2, 0, Qt.AlignLeft)
        dft_bottom_layout.addWidget(self.label_mag2, 3, 0)
        dft_bottom_layout.addWidget(QLabel("Phase 2:"), 2, 1, Qt.AlignLeft)
        dft_bottom_layout.addWidget(self.label_phase2, 3, 1)
        dft_bottom_layout.addWidget(QLabel("Combined Result:"), 0, 2, 1, 1, Qt.AlignLeft)
        dft_bottom_layout.addWidget(self.label_dft_result, 1, 2, 3, 1)
        
        # ---- Harris Mode Bottom Panel ----
        harris_bottom_widget = QWidget()
        harris_bottom_layout = QGridLayout(harris_bottom_widget)
        harris_bottom_layout.setSpacing(10)
        
        # Image labels for Harris detection
        self.label_harris_img1 = ClickableQLabel("Harris Corners 1")
        self.label_harris_response1 = ClickableQLabel("Response Map 1")
        self.label_harris_img2 = ClickableQLabel("Harris Corners 2")
        self.label_harris_response2 = ClickableQLabel("Response Map 2")
        
        # Set alignment and styling
        for label in [self.label_harris_img1, self.label_harris_response1, 
                      self.label_harris_img2,self.label_harris_response2
]:
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(275, 240)
            label.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        
        # Add to grid
        harris_bottom_layout.addWidget(QLabel("Corners 1:"), 0, 0, Qt.AlignLeft)
        harris_bottom_layout.addWidget(self.label_harris_img1, 1, 0)
        harris_bottom_layout.addWidget(QLabel("Response 1:"), 0, 1, Qt.AlignLeft)
        harris_bottom_layout.addWidget(self.label_harris_response1, 1, 1)
        harris_bottom_layout.addWidget(QLabel("Corners 2:"), 2, 0, Qt.AlignLeft)
        harris_bottom_layout.addWidget(self.label_harris_img2, 3, 0)
        harris_bottom_layout.addWidget(QLabel("Response 2:"), 2, 1, Qt.AlignLeft)
        harris_bottom_layout.addWidget(self.label_harris_response2, 3, 1)
        
        # Add both panels to stacked widget
        self.bottom_stacked.addWidget(dft_bottom_widget)
        self.bottom_stacked.addWidget(harris_bottom_widget)
        
        # Add bottom group box to wrap the stacked widget
        bottom_group = QGroupBox("Analysis Results")
        bottom_group.setStyleSheet("QGroupBox {font-weight: bold; font-size: 14px;}")
        bottom_main_layout = QVBoxLayout()
        bottom_main_layout.addWidget(self.bottom_stacked)
        bottom_group.setLayout(bottom_main_layout)
        
        # Add image panels to left layout
        left_layout.addWidget(top_group, 1)
        left_layout.addWidget(bottom_group, 2)
        
        # ========= RIGHT PANEL (CONTROLS) =========
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Image loading buttons with icons
        load_buttons_layout = QHBoxLayout()
        
        self.btn_load1 = QPushButton("  Load Image 1")
        self.btn_load1.setIcon(QIcon.fromTheme("document-open"))
        self.btn_load1.setIconSize(QSize(20, 20))
        self.btn_load1.setMinimumHeight(40)
        self.btn_load1.clicked.connect(self.load_image1)
        
        self.btn_load2 = QPushButton("  Load Image 2")
        self.btn_load2.setIcon(QIcon.fromTheme("document-open"))
        self.btn_load2.setIconSize(QSize(20, 20))
        self.btn_load2.setMinimumHeight(40)
        self.btn_load2.clicked.connect(self.load_image2)
        
        load_buttons_layout.addWidget(self.btn_load1)
        load_buttons_layout.addWidget(self.btn_load2)
        self.label_harris_img1.clicked.connect(self.on_harris_image1_clicked)
        self.label_harris_response1.clicked.connect(self.on_label_harris_response1_clicked)
        self.label_harris_response2.clicked.connect(self.on_label_harris_response2_clicked)
        self.label_harris_img2.clicked.connect(self.on_harris_image2_clicked)
        
        # Create stacked widget for right panel controls based on mode
        self.controls_stacked = QStackedWidget()
        
        # ---- DFT Controls ----
        dft_controls = QWidget()
        dft_controls_layout = QVBoxLayout(dft_controls)
        dft_controls_layout.setSpacing(15)
        
        # Create a tabbed interface for different reconstruction modes
        reconstruction_tabs = QTabWidget()
        
        # ---- Tab 1: Standard Combination ----
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        
        # Radio buttons for combination mode
        self.radio_mode1 = QRadioButton("Mag 1 + Phase 2")
        self.radio_mode1.setChecked(True)
        self.radio_mode2 = QRadioButton("Mag 2 + Phase 1")
        
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.radio_mode1)
        mode_layout.addWidget(self.radio_mode2)
        
        # Create color scheme selection
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Color Scheme:"))
        
        self.color_scheme_combo = QComboBox()
        self.color_scheme_combo.addItems(["Default", "Vibrant", "Grayscale", "Thermal", "Ocean"])
        self.color_scheme_combo.currentIndexChanged.connect(self.update_combined_result)
        
        color_layout.addWidget(self.color_scheme_combo)
        
        # Button to perform the combination
        self.btn_combine = QPushButton("Combine Magnitude & Phase")
        self.btn_combine.setMinimumHeight(40)
        self.btn_combine.setEnabled(False)
        self.btn_combine.clicked.connect(self.update_combined_result)
        
        # Add all elements to tab1 layout
        tab1_layout.addLayout(mode_layout)
        tab1_layout.addLayout(color_layout)
        tab1_layout.addWidget(self.btn_combine)
        
        # ---- Tab 2: Custom Reconstruction ----
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        
        # Source image selection
        self.radio_src1 = QRadioButton("Use Image 1")
        self.radio_src1.setChecked(True)
        self.radio_src2 = QRadioButton("Use Image 2")
        
        src_layout = QHBoxLayout()
        src_layout.addWidget(self.radio_src1)
        src_layout.addWidget(self.radio_src2)
        
        # Component checkboxes
        component_layout = QHBoxLayout()
        
        self.check_use_mag = QCheckBox("Use Magnitude")
        self.check_use_mag.setChecked(True)
        self.check_use_phase = QCheckBox("Use Phase")
        self.check_use_phase.setChecked(True)
        
        component_layout.addWidget(self.check_use_mag)
        component_layout.addWidget(self.check_use_phase)
        
        # Magnitude percentage slider
        mag_layout = QHBoxLayout()
        mag_layout.addWidget(QLabel("Magnitude %:"))
        self.slider_mag = QSlider(Qt.Horizontal)
        self.slider_mag.setRange(0, 100)
        self.slider_mag.setValue(100)
        self.label_mag_percent = QLabel("100%")
        self.slider_mag.valueChanged.connect(self.update_mag_label)
        mag_layout.addWidget(self.slider_mag)
        mag_layout.addWidget(self.label_mag_percent)
        
        # Phase percentage slider
        phase_layout = QHBoxLayout()
        phase_layout.addWidget(QLabel("Phase %:"))
        self.slider_phase = QSlider(Qt.Horizontal)
        self.slider_phase.setRange(0, 100)
        self.slider_phase.setValue(100)
        self.label_phase_percent = QLabel("100%")
        self.slider_phase.valueChanged.connect(self.update_phase_label)
        phase_layout.addWidget(self.slider_phase)
        phase_layout.addWidget(self.label_phase_percent)
        
        # Predefined reconstruction types
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Custom Settings", 
            "100% Magnitude Only", 
            "100% Phase Only", 
            "50% Magnitude + 50% Phase",
            "80% Magnitude + 20% Phase",
            "20% Magnitude + 80% Phase"
        ])
        self.preset_combo.currentIndexChanged.connect(self.apply_preset)
        
        preset_layout.addWidget(self.preset_combo)
        
        # Reconstruction button
        self.btn_reconstruct = QPushButton("Reconstruct Image")
        self.btn_reconstruct.setMinimumHeight(40)
        self.btn_reconstruct.setEnabled(False)
        self.btn_reconstruct.clicked.connect(self.update_custom_reconstruction)
        
        # Add all elements to tab2 layout
        tab2_layout.addLayout(src_layout)
        tab2_layout.addLayout(component_layout)
        tab2_layout.addLayout(mag_layout)
        tab2_layout.addLayout(phase_layout)
        tab2_layout.addLayout(preset_layout)
        tab2_layout.addWidget(self.btn_reconstruct)
        
        # Add tabs to tabbed interface
        reconstruction_tabs.addTab(tab1, "Standard Combination")
        reconstruction_tabs.addTab(tab2, "Custom Reconstruction")
        
        # Add all controls elements to main DFT controls layout
        dft_controls_layout.addWidget(reconstruction_tabs)
        dft_controls_layout.addStretch(1)
        
        # ---- Harris Controls ----
        harris_controls = QWidget()
        harris_controls_layout = QVBoxLayout(harris_controls)
        harris_controls_layout.setSpacing(15)
        
        # Group for Harris Parameters
        params_group = QGroupBox("Harris Parameters")
        params_layout = QFormLayout()
        
        # Block size parameter
        self.block_size_spin = QSpinBox()
        self.block_size_spin.setRange(3,7)
        self.block_size_spin.setValue(3)
        self.block_size_spin.setSingleStep(2)
        params_layout.addRow("Block Size:", self.block_size_spin)
        
        # K parameter
        self.k_param_spin = QDoubleSpinBox()
        self.k_param_spin.setRange(0.01, 0.2)
        self.k_param_spin.setValue(0.04)
        self.k_param_spin.setSingleStep(0.01)
        self.k_param_spin.setDecimals(3)
        params_layout.addRow("K Parameter:", self.k_param_spin)
        
        # Threshold parameter
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.001, 0.5)
        self.threshold_spin.setValue(0.01)
        self.threshold_spin.setSingleStep(0.005)
        self.threshold_spin.setDecimals(3)
        params_layout.addRow("Threshold (ratio):", self.threshold_spin)
        
        params_group.setLayout(params_layout)
        
        # Button for analyzing with Harris
        self.btn_apply_harris1 = QPushButton("Apply Harris to Image 1")
        self.btn_apply_harris1.setMinimumHeight(40)
        self.btn_apply_harris1.setEnabled(False)
        self.btn_apply_harris1.clicked.connect(self.apply_harris_to_image1)
        
        self.btn_apply_harris2 = QPushButton("Apply Harris to Image 2")
        self.btn_apply_harris2.setMinimumHeight(40)
        self.btn_apply_harris2.setEnabled(False)
        self.btn_apply_harris2.clicked.connect(self.apply_harris_to_image2)
        
        # Add controls to harris layout
        harris_controls_layout.addWidget(params_group)
        harris_controls_layout.addWidget(self.btn_apply_harris1)
        harris_controls_layout.addWidget(self.btn_apply_harris2)
        harris_controls_layout.addStretch(1)
        
        # Add both control widgets to stacked widget
        self.controls_stacked.addWidget(dft_controls)
        self.controls_stacked.addWidget(harris_controls)
        
        # Controls Group to wrap the stacked widget
        controls_group = QGroupBox("Controls")
        controls_group.setStyleSheet("QGroupBox {font-weight: bold; font-size: 14px;}")
        controls_main_layout = QVBoxLayout()
        controls_main_layout.addLayout(load_buttons_layout)
        controls_main_layout.addWidget(self.controls_stacked)
        controls_group.setLayout(controls_main_layout)
        
        # Feedback & Interpretation Group
        interpretation_group = QGroupBox("Analysis & Interpretation")
        interpretation_group.setStyleSheet("QGroupBox {font-weight: bold; font-size: 14px;}")
        interpretation_layout = QVBoxLayout()
        
        self.interpretation_tabs = QTabWidget()
        
        # Create tabs for different interpretations
        self.image1_tab = QTextEdit()
        self.image1_tab.setReadOnly(True)
        self.image2_tab = QTextEdit()
        self.image2_tab.setReadOnly(True)
        self.combined_tab = QTextEdit()
        self.combined_tab.setReadOnly(True)
        self.metrics_tab = QTextEdit()
        self.metrics_tab.setReadOnly(True)
        
        self.interpretation_tabs.addTab(self.image1_tab, "Image 1")
        self.interpretation_tabs.addTab(self.image2_tab, "Image 2")
        self.interpretation_tabs.addTab(self.combined_tab, "Combined")
        self.interpretation_tabs.addTab(self.metrics_tab, "Metrics")
        
        interpretation_layout.addWidget(self.interpretation_tabs)
        interpretation_group.setLayout(interpretation_layout)
        
        # Add all groups to right panel
        right_layout.addWidget(controls_group)
        right_layout.addWidget(interpretation_group, 1)
        
        # Add panels to the main splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        
        # Set the ratio of left panel to right panel (70:30)
        main_splitter.setSizes([900, 400])
        
        # Add splitter to main layout
        main_layout.addWidget(main_splitter)
        
        self.setLayout(main_layout)
        
        # Connect signals for the radio buttons
        self.radio_mode1.toggled.connect(self.update_combined_result)
        self.radio_mode2.toggled.connect(self.update_combined_result)
    
    def switch_mode(self, index):
        """Switch between DFT and Harris modes"""
        # Switch the stacked widgets
        self.bottom_stacked.setCurrentIndex(index)
        self.controls_stacked.setCurrentIndex(index)
        
        # Update interpretation text based on mode
        if index == 0:  # DFT mode
            self.interpretation_tabs.setTabText(0, "Image 1")
            self.interpretation_tabs.setTabText(1, "Image 2")
            self.interpretation_tabs.setTabText(2, "Combined")
            
            # Reload DFT analyses if available
            if self.img1 is not None and self.img1_dft_components is not None:
                self.update_image1_analysis()
            if self.img2 is not None and self.img2_dft_components is not None:
                self.update_image2_analysis()
        else:  # Harris mode
            self.interpretation_tabs.setTabText(0, "Harris 1")
            self.interpretation_tabs.setTabText(1, "Harris 2")
            self.interpretation_tabs.setTabText(2, "Notes")
            self.combined_tab.setText("Harris corner detection identifies points where the image gradient has significant changes in multiple directions.\n\nAdjust parameters to optimize detection:\n- Block size: Neighborhood size for corner detection\n- k: Harris detector sensitivity parameter (usually 0.04-0.06)\n- Threshold: Minimum response value for corner detection")
    
    def apply_styles(self):
        """Apply modern styling to the application"""
        # Set application style sheet
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                color: #333333;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QGroupBox {
                border: 1px solid #cccccc;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 10px;
            }
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a76d8;
            }
            QPushButton:pressed {
                background-color: #2a66c8;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }
            QLabel {
                color: #333333;
                font-size: 12px;
            }
            QTextEdit {
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: #ffffff;
                padding: 4px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QRadioButton {
                spacing: 8px;
                font-size: 13px;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
            }
            QComboBox {
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 24px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #cccccc;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4a86e8;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #3a76d8;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 5px;
            }
            QTabBar::tab {
                background: #e0e0e0;
                border: 1px solid #cccccc;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 5px 10px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #4a86e8;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background: #d0d0d0;
            }
            QSpinBox, QDoubleSpinBox {
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 4px;
                min-height: 20px;
            }
        """)
    
    def update_mag_label(self):
        """Update magnitude percentage label"""
        value = self.slider_mag.value()
        self.label_mag_percent.setText(f"{value}%")
    
    def update_phase_label(self):
        """Update phase percentage label"""
        value = self.slider_phase.value()
        self.label_phase_percent.setText(f"{value}%")
    
    def apply_preset(self):
        """Apply preset reconstruction parameters"""
        preset = self.preset_combo.currentText()
        
        if preset == "100% Magnitude Only":
            self.check_use_mag.setChecked(True)
            self.check_use_phase.setChecked(False)
            self.slider_mag.setValue(100)
            self.slider_phase.setValue(0)
            
        elif preset == "100% Phase Only":
            self.check_use_mag.setChecked(False)
            self.check_use_phase.setChecked(True)
            self.slider_mag.setValue(0)
            self.slider_phase.setValue(100)
            
        elif preset == "50% Magnitude + 50% Phase":
            self.check_use_mag.setChecked(True)
            self.check_use_phase.setChecked(True)
            self.slider_mag.setValue(50)
            self.slider_phase.setValue(50)
            
        elif preset == "80% Magnitude + 20% Phase":
            self.check_use_mag.setChecked(True)
            self.check_use_phase.setChecked(True)
            self.slider_mag.setValue(80)
            self.slider_phase.setValue(20)
            
        elif preset == "20% Magnitude + 80% Phase":
            self.check_use_mag.setChecked(True)
            self.check_use_phase.setChecked(True)
            self.slider_mag.setValue(20)
            self.slider_phase.setValue(80)
            
        # Custom settings - do nothing
    
    def load_image1(self):
        """Load the first image"""
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open Image 1", "", "Image files (*.jpg *.png *.bmp)"
        )
        if fname:
            self.img1 = cv2.imread(fname)
            
            # Display the original image
            self.label_img1.setPixmap(np_to_pixmap(self.img1).scaled(
                self.label_img1.width(), self.label_img1.height(), 
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            
            # Enable Harris button
            self.btn_apply_harris1.setEnabled(True)
            
            # Update analysis based on current mode
            if self.mode_combo.currentIndex() == 0:  # DFT mode
                self.update_image1_analysis()
            else:  # Harris mode
                self.apply_harris_to_image1()
    
    def load_image2(self):
        """Load the second image"""
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open Image 2", "", "Image files (*.jpg *.png *.bmp)"
        )
        if fname:
            self.img2 = cv2.imread(fname)
            
            # Display the original image
            self.label_img2.setPixmap(np_to_pixmap(self.img2).scaled(
                self.label_img2.width(), self.label_img2.height(), 
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            
            # Enable Harris button
            self.btn_apply_harris2.setEnabled(True)
            
            # Update analysis based on current mode
            if self.mode_combo.currentIndex() == 0:  # DFT mode
                self.update_image2_analysis()
            else:  # Harris mode
                self.apply_harris_to_image2()
    
    def update_image1_analysis(self):
        """Update the first image DFT analysis"""
        if self.img1 is not None:
            # Perform DFT analysis
            mag_img, phase_img, interpretation, dft_components = dft_analysis(self.img1)
            
            # Store DFT components for later interpolation
            self.img1_dft_components = dft_components
            
            # Display the DFT spectra
            self.label_mag1.setPixmap(np_to_pixmap(mag_img).scaled(
                self.label_mag1.width(), self.label_mag1.height(), 
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            self.label_phase1.setPixmap(np_to_pixmap(phase_img).scaled(
                self.label_phase1.width(), self.label_phase1.height(), 
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            
            # Update interpretation text
            self.image1_tab.setText(interpretation)
            self.interpretation_tabs.setCurrentIndex(0)
            
            # Enable buttons if needed
            if self.img2 is not None and self.img2_dft_components is not None:
                self.btn_combine.setEnabled(True)
                self.btn_reconstruct.setEnabled(True)
                # Auto-update combined result if both images are loaded
                self.update_combined_result()
            else:
                # Enable reconstruction button for single-image operations
                self.btn_reconstruct.setEnabled(True)
    
    def update_image2_analysis(self):
        """Update the second image DFT analysis"""
        if self.img2 is not None:
            # Perform DFT analysis
            mag_img, phase_img, interpretation, dft_components = dft_analysis(self.img2)
            
            # Store DFT components for later interpolation
            self.img2_dft_components = dft_components
            
            # Display the DFT spectra
            self.label_mag2.setPixmap(np_to_pixmap(mag_img).scaled(
                self.label_mag2.width(), self.label_mag2.height(), 
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            self.label_phase2.setPixmap(np_to_pixmap(phase_img).scaled(
                self.label_phase2.width(), self.label_phase2.height(), 
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            
            # Update interpretation text
            self.image2_tab.setText(interpretation)
            self.interpretation_tabs.setCurrentIndex(1)
            
            # Enable buttons if needed
            if self.img1 is not None and self.img1_dft_components is not None:
                self.btn_combine.setEnabled(True)
                self.btn_reconstruct.setEnabled(True)
                # Auto-update combined result if both images are loaded
                self.update_combined_result()
            else:
                # Enable reconstruction button for single-image operations
                self.btn_reconstruct.setEnabled(True)
    
    def apply_harris_to_image1(self):
        """Apply Harris corner detection to image 1"""
        if self.img1 is None:
            return
        
        # Get parameters
        block_size = self.block_size_spin.value()
        k_param = self.k_param_spin.value()
        threshold_ratio = self.threshold_spin.value()
        
        # Apply Harris corner detection
        response_vis,corner_img, num_corners, interpretation = harris_corner_detector(
            self.img1, block_size, k_param, threshold_ratio
        )
        
        
        # Display results
        self.label_harris_img1.setPixmap(np_to_pixmap(corner_img).scaled(
            self.label_harris_img1.width(), self.label_harris_img1.height(), 
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        
        self.label_harris_response1.setPixmap(np_to_pixmap(response_vis).scaled(
            self.label_harris_response1.width(), self.label_harris_response1.height(), 
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        
        # Update interpretation
        self.image1_tab.setText(interpretation)
        self.interpretation_tabs.setCurrentIndex(0)
        self.corner_img1=corner_img
        self.response_vis1=response_vis

    
    def apply_harris_to_image2(self):
        """Apply Harris corner detection to image 2"""
        if self.img2 is None:
            return
        
        # Get parameters
        block_size = self.block_size_spin.value()
        k_param = self.k_param_spin.value()
        threshold_ratio = self.threshold_spin.value()
        
        # Apply Harris corner detection
        corner_img, response_vis, num_corners, interpretation = harris_corner_detector(
            self.img2, block_size, k_param, threshold_ratio
        )
        
        # Display results
        self.label_harris_img2.setPixmap(np_to_pixmap(corner_img).scaled(
            self.label_harris_img2.width(), self.label_harris_img2.height(), 
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        
        self.label_harris_response2.setPixmap(np_to_pixmap(response_vis).scaled(
            self.label_harris_response2.width(), self.label_harris_response2.height(), 
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        self.corner_img2=corner_img
        self.response_vis2=response_vis
        # Update interpretation
        self.image2_tab.setText(interpretation)
        self.interpretation_tabs.setCurrentIndex(1)
    
    def update_combined_result(self):
        """Update the combined result based on selected options"""
        if self.img1 is None or self.img2 is None or self.img1_dft_components is None or self.img2_dft_components is None:
            return
        
        # Determine which combination mode to use
        if self.radio_mode1.isChecked():
            # Use magnitude from first image and phase from second image
            magnitude_components = self.img1_dft_components
            phase_components = self.img2_dft_components
            original_image = self.img1  # For metrics calculation
            combination_text = "Magnitude from Image 1 + Phase from Image 2"
        else:
            # Use magnitude from second image and phase from first image
            magnitude_components = self.img2_dft_components
            phase_components = self.img1_dft_components
            original_image = self.img2  # For metrics calculation
            combination_text = "Magnitude from Image 2 + Phase from Image 1"
        
        # Combine magnitude and phase
        interpolated_img, interpretation = combine_magnitude_phase(
            magnitude_components, phase_components
        )
        
        # Apply color scheme based on selection
        color_scheme = self.color_scheme_combo.currentText()
        if color_scheme == "Vibrant":
            interpolated_img = cv2.applyColorMap(cv2.cvtColor(interpolated_img, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
        elif color_scheme == "Grayscale":
            gray = cv2.cvtColor(interpolated_img, cv2.COLOR_BGR2GRAY)
            interpolated_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif color_scheme == "Thermal":
            interpolated_img = cv2.applyColorMap(cv2.cvtColor(interpolated_img, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_HOT)
        elif color_scheme == "Ocean":
            interpolated_img = cv2.applyColorMap(cv2.cvtColor(interpolated_img, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_OCEAN)
        
        # Store the result
        self.combined_img = interpolated_img
        
        # Display the result
        self.label_dft_result.setPixmap(np_to_pixmap(interpolated_img).scaled(
            self.label_dft_result.width(), self.label_dft_result.height(), 
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        
        # Update interpretation with combination details
        combined_interpretation = f"Current Combination: {combination_text}\n\n{interpretation}"
        self.combined_tab.setText(combined_interpretation)
        self.interpretation_tabs.setCurrentIndex(2)
        
        # Calculate and display metrics
        if self.combined_img is not None and original_image is not None:
            self.update_metrics(original_image, self.combined_img)
    
    def update_custom_reconstruction(self):
        """Update the custom reconstruction based on sliders and checkboxes"""
        # Determine which image to use
        if self.radio_src1.isChecked() and self.img1_dft_components is not None:
            dft_components = self.img1_dft_components
            original_image = self.img1
            source_text = "Image 1"
        elif self.radio_src2.isChecked() and self.img2_dft_components is not None:
            dft_components = self.img2_dft_components
            original_image = self.img2
            source_text = "Image 2"
        else:
            QMessageBox.warning(self, "Warning", "Please load at least one image first.")
            return
        
        # Get component settings
        use_magnitude = self.check_use_mag.isChecked()
        use_phase = self.check_use_phase.isChecked()
        mag_weight = self.slider_mag.value() / 100.0
        phase_weight = self.slider_phase.value() / 100.0
        
        # Determine colormap from the color scheme combobox
        color_scheme = self.color_scheme_combo.currentText()
        if color_scheme == "Vibrant":
            colormap = cv2.COLORMAP_JET
        elif color_scheme == "Grayscale":
            colormap = None  # Special case for grayscale
        elif color_scheme == "Thermal":
            colormap = cv2.COLORMAP_HOT
        elif color_scheme == "Ocean":
            colormap = cv2.COLORMAP_OCEAN
        else:
            colormap = cv2.COLORMAP_BONE  # Default
        
        # Reconstruct image with custom settings
        reconstructed_img, interpretation = reconstruct_custom_image(
            dft_components,
            use_magnitude=use_magnitude,
            use_phase=use_phase,
            mag_weight=mag_weight,
            phase_weight=phase_weight,
            colormap=colormap
        )
        
        # Store and display the result
        self.combined_img = reconstructed_img
        
        self.label_dft_result.setPixmap(np_to_pixmap(reconstructed_img).scaled(
            self.label_dft_result.width(), self.label_dft_result.height(), 
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        
        # Update interpretation text
        settings_description = f"""Reconstruction Settings:
- Source: {source_text}
- {"Using" if use_magnitude else "Not using"} magnitude component {f"({int(mag_weight*100)}%)" if use_magnitude else ""}
- {"Using" if use_phase else "Not using"} phase component {f"({int(phase_weight*100)}%)" if use_phase else ""}
- Color scheme: {color_scheme}
        
{interpretation}"""
        
        self.combined_tab.setText(settings_description)
        self.interpretation_tabs.setCurrentIndex(2)
        
        # Calculate and display metrics
        self.update_metrics(original_image, reconstructed_img)
    
    def update_metrics(self, original_img, reconstructed_img):
        """Calculate and display metrics comparing original and reconstructed images"""
        try:
            # Calculate metrics
            metrics = calculate_reconstruction_metrics(original_img, reconstructed_img)
            
            # Format metrics text
            metrics_text = f"""Reconstruction Quality Metrics:

Mean Squared Error (MSE): {metrics['MSE']:.2f}
- Lower values indicate better reconstruction
- MSE = 0 means perfect reconstruction

Peak Signal-to-Noise Ratio (PSNR): {metrics['PSNR']:.2f} dB
- Higher values indicate better quality
- Typical values: 20-40 dB (higher is better)
- PSNR > 30 dB suggests good quality

Structural Similarity Index (SSIM): {metrics['SSIM']:.4f}
- Range: 0.0 (no similarity) to 1.0 (identical images)
- Values above 0.9 indicate high similarity

Interpretation:
- {'Good reconstruction quality' if metrics['PSNR'] > 30 else 'Moderate reconstruction quality' if metrics['PSNR'] > 20 else 'Poor reconstruction quality'}
- {'High structural similarity' if metrics['SSIM'] > 0.8 else 'Moderate structural similarity' if metrics['SSIM'] > 0.5 else 'Low structural similarity'}
"""
            
            # Update the metrics tab
            self.metrics_tab.setText(metrics_text)
            
        except Exception as e:
            self.metrics_tab.setText(f"Error calculating metrics: {str(e)}")

    def on_harris_image1_clicked(self):
        if(not self.corner_img1 is None):
            cv2.imshow('Corner Response 1', self.corner_img1)
    def on_harris_image2_clicked(self):
        if(not self.corner_img2 is None):
            cv2.imshow('Corner Response 2', self.corner_img2)
    def on_label_harris_response1_clicked(self):
        if(not self.response_vis1 is None):
            cv2.imshow('Response 1', self.response_vis1)
    def on_label_harris_response2_clicked(self):
        if(not self.response_vis2 is None):
            cv2.imshow('Response 2', self.response_vis2)
            