import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon, QPalette, QColor
from PyQt5.QtCore import Qt

# Import the ImageProcessorClass class from the separate file
from imageProcessor_class import ImageProcessorClass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Set application style and palette for a modern look
    app.setStyle('Fusion')
    
    # Create a custom dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.WindowText, QColor(51, 51, 51))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(51, 51, 51))
    palette.setColor(QPalette.Text, QColor(51, 51, 51))
    palette.setColor(QPalette.Button, QColor(74, 134, 232))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    # Create and show the application window
    window = ImageProcessorClass()
    window.show()
    
    # Execute the application
    sys.exit(app.exec_())