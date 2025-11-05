"""Ventana principal de la aplicación PyQt5"""
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget,
    QStatusBar, QMessageBox
)
from PyQt5.QtCore import QTimer
from frontend_desktop.views.training_view import TrainingView
from frontend_desktop.views.inference_view import InferenceView
from frontend_desktop.views.results_view import ResultsView
from frontend_desktop.utils.api_client import APIClient


class MainWindow(QMainWindow):
    """Ventana principal con tabs"""
    
    def __init__(self):
        super().__init__()
        self.api_client = APIClient()
        self.init_ui()
        self.check_backend()
    
    def init_ui(self):
        """Inicializa la interfaz"""
        self.setWindowTitle("🌾 U-Net Anomaly Detection System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Tabs
        self.tabs = QTabWidget()
        
        # Vistas
        self.training_view = TrainingView(self.api_client)
        self.inference_view = InferenceView(self.api_client)
        self.results_view = ResultsView(self.api_client)
        
        self.tabs.addTab(self.training_view, "📚 Entrenar Modelo")
        self.tabs.addTab(self.inference_view, "🔍 Inferencia")
        self.tabs.addTab(self.results_view, "📊 Resultados")
        
        layout.addWidget(self.tabs)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Listo")
    
    def check_backend(self):
        """Verifica si el backend está disponible"""
        if not self.api_client.health_check():
            QMessageBox.warning(
                self,
                "Backend no disponible",
                "No se puede conectar al backend.\n\n"
                "Asegúrate de que el servidor esté corriendo:\n"
                "python -m uvicorn backend.main:app --reload"
            )
            self.status_bar.showMessage("⚠️ Backend desconectado", 5000)
        else:
            self.status_bar.showMessage("✅ Conectado al backend", 3000)
