"""Aplicación principal PyQt5"""
import sys
from PyQt5.QtWidgets import QApplication
from frontend_desktop.views.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
