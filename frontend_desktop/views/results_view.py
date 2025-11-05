"""Vista de resultados"""
from PyQt5.QtWidgets import *


class ResultsView(QWidget):
    def __init__(self, api_client):
        super().__init__()
        self.api_client = api_client
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Modelos entrenados
        models_group = QGroupBox("🤖 Modelos Entrenados")
        models_layout = QVBoxLayout()
        
        self.models_table = QTableWidget()
        self.models_table.setColumnCount(5)
        self.models_table.setHorizontalHeaderLabels(["ID", "Backbone", "IoU", "Epochs", "Fecha"])
        self.models_table.horizontalHeader().setStretchLastSection(True)
        
        btn_refresh = QPushButton("🔄 Actualizar")
        btn_refresh.clicked.connect(self.load_models)
        
        models_layout.addWidget(self.models_table)
        models_layout.addWidget(btn_refresh)
        models_group.setLayout(models_layout)
        layout.addWidget(models_group)
        
        # Info
        info_label = QLabel("📊 Aquí puedes ver todos tus modelos entrenados y sus métricas.")
        layout.addWidget(info_label)
        
        # Cargar modelos
        self.load_models()
    
    def load_models(self):
        try:
            models = self.api_client.list_models()
            self.models_table.setRowCount(len(models))
            
            for i, model in enumerate(models):
                self.models_table.setItem(i, 0, QTableWidgetItem(model["model_id"]))
                self.models_table.setItem(i, 1, QTableWidgetItem(model.get("backbone", "N/A")))
                self.models_table.setItem(i, 2, QTableWidgetItem(f"{model.get('final_iou', 0):.4f}"))
                self.models_table.setItem(i, 3, QTableWidgetItem(str(model.get("epochs_trained", 0))))
                self.models_table.setItem(i, 4, QTableWidgetItem(model["created_at"][:10]))
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"No se pudieron cargar los modelos:\n{str(e)}")
