"""Vista de inferencia"""
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer
import os


class InferenceView(QWidget):
    def __init__(self, api_client):
        super().__init__()
        self.api_client = api_client
        self.current_job_id = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Config
        config_group = QGroupBox("🔍 Configuración de Inferencia")
        config_layout = QVBoxLayout()
        
        # Imagen
        img_layout = QHBoxLayout()
        self.image_input = QLineEdit()
        self.image_input.setPlaceholderText("Archivo de imagen .tif")
        btn_browse = QPushButton("📁 Buscar")
        btn_browse.clicked.connect(self.browse_image)
        img_layout.addWidget(QLabel("Imagen:"))
        img_layout.addWidget(self.image_input)
        img_layout.addWidget(btn_browse)
        config_layout.addLayout(img_layout)
        
        # Modelo
        model_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        btn_refresh = QPushButton("🔄")
        btn_refresh.clicked.connect(self.load_models)
        model_layout.addWidget(QLabel("Modelo:"))
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(btn_refresh)
        config_layout.addLayout(model_layout)
        
        # Parámetros
        params_layout = QHBoxLayout()
        
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.5)
        params_layout.addWidget(QLabel("Umbral:"))
        params_layout.addWidget(self.threshold_spin)
        
        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(32, 512)
        self.stride_spin.setValue(256)
        params_layout.addWidget(QLabel("Stride:"))
        params_layout.addWidget(self.stride_spin)
        
        config_layout.addLayout(params_layout)
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Botones
        actions_layout = QHBoxLayout()
        self.btn_predict = QPushButton("🎯 PREDECIR ANOMALÍAS")
        self.btn_predict.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-size: 14px; padding: 10px; }")
        self.btn_predict.clicked.connect(self.start_prediction)
        actions_layout.addWidget(self.btn_predict)
        layout.addLayout(actions_layout)
        
        # Progreso
        progress_group = QGroupBox("📊 Progreso")
        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Listo")
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Resultados
        results_group = QGroupBox("📈 Resultados")
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(150)
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Cargar modelos al inicio
        self.load_models()
    
    def browse_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "Seleccionar Imagen", "", "TIFF Files (*.tif *.tiff)")
        if file:
            self.image_input.setText(file)
    
    def load_models(self):
        try:
            models = self.api_client.list_models()
            self.model_combo.clear()
            for model in models:
                self.model_combo.addItem(f"{model['model_id']} (IoU: {model.get('final_iou', 0):.3f})", model['model_id'])
            self.results_text.append(f"✅ {len(models)} modelos cargados")
        except Exception as e:
            self.results_text.append(f"❌ Error cargando modelos: {str(e)}")
    
    def start_prediction(self):
        image_path = self.image_input.text()
        if not image_path or not os.path.exists(image_path):
            QMessageBox.warning(self, "Error", "Selecciona una imagen válida")
            return
        
        model_id = self.model_combo.currentData()
        if not model_id:
            QMessageBox.warning(self, "Error", "Selecciona un modelo")
            return
        
        try:
            response = self.api_client.start_prediction(
                image_path=image_path,
                model_id=model_id,
                threshold=self.threshold_spin.value(),
                stride=self.stride_spin.value(),
            )
            
            self.current_job_id = response["job_id"]
            self.results_text.append(f"✅ Predicción iniciada: {self.current_job_id}")
            self.btn_predict.setEnabled(False)
            self.timer.start(1000)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo iniciar predicción:\n{str(e)}")
            self.results_text.append(f"❌ Error: {str(e)}")
    
    def update_progress(self):
        if not self.current_job_id:
            return
        
        try:
            status = self.api_client.get_inference_status(self.current_job_id)
            progress = int(status["progress"] * 100)
            self.progress_bar.setValue(progress)
            self.status_label.setText(f"{status['status']} - {progress}%")
            
            if status["status"] in ["completed", "failed"]:
                self.timer.stop()
                self.btn_predict.setEnabled(True)
                
                if status["status"] == "completed":
                    stats = status.get("stats", {})
                    result_text = (
                        f"\n✅ Predicción completada!\n"
                        f"Archivo: {status['output_path']}\n"
                        f"Píxeles anómalos: {stats.get('anomaly_percentage', 0):.2f}%"
                    )
                    self.results_text.append(result_text)
                    QMessageBox.information(self, "Completado", result_text)
                elif status["status"] == "failed":
                    self.results_text.append(f"❌ Error: {status.get('error_message')}")
        except Exception as e:
            self.results_text.append(f"⚠️ Error: {str(e)}")
