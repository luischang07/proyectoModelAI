"""Vista de entrenamiento"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QSpinBox, QComboBox,
    QProgressBar, QTextEdit, QFileDialog, QMessageBox
)
from PyQt5.QtCore import QTimer, pyqtSignal
import os


class TrainingView(QWidget):
    """Vista para configurar y entrenar modelos"""
    
    training_started = pyqtSignal(str)  # job_id
    
    def __init__(self, api_client):
        super().__init__()
        self.api_client = api_client
        self.current_job_id = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.init_ui()
    
    def init_ui(self):
        """Inicializa la UI"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # === TIPO DE ENTRENAMIENTO ===
        type_group = QGroupBox("🎯 Tipo de Entrenamiento")
        type_layout = QVBoxLayout()
        
        self.training_type_combo = QComboBox()
        self.training_type_combo.addItems([
            "Supervisado (con máscaras)",
            "No Supervisado (sin máscaras - Autoencoder)"
        ])
        self.training_type_combo.currentIndexChanged.connect(self.on_training_type_changed)
        
        type_help = QLabel(
            "• <b>Supervisado</b>: Requiere imágenes + máscaras etiquetadas\n"
            "• <b>No Supervisado</b>: Solo requiere imágenes (aprende patrones normales)"
        )
        type_help.setStyleSheet("color: #666; font-size: 10px;")
        
        type_layout.addWidget(self.training_type_combo)
        type_layout.addWidget(type_help)
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)
        
        # === CONFIGURACIÓN ===
        config_group = QGroupBox("⚙️ Configuración")
        config_layout = QVBoxLayout()
        
        # Carpetas
        folder_layout = QHBoxLayout()
        self.images_folder_input = QLineEdit()
        self.images_folder_input.setPlaceholderText("Carpeta con imágenes .tif")
        btn_browse_images = QPushButton("📁 Buscar")
        btn_browse_images.clicked.connect(lambda: self.browse_folder(self.images_folder_input))
        folder_layout.addWidget(QLabel("Imágenes:"))
        folder_layout.addWidget(self.images_folder_input)
        folder_layout.addWidget(btn_browse_images)
        config_layout.addLayout(folder_layout)
        
        mask_layout = QHBoxLayout()
        self.masks_folder_input = QLineEdit()
        self.masks_folder_input.setPlaceholderText("Carpeta con máscaras .tif")
        btn_browse_masks = QPushButton("📁 Buscar")
        btn_browse_masks.clicked.connect(lambda: self.browse_folder(self.masks_folder_input))
        self.mask_label = QLabel("Máscaras:")
        mask_layout.addWidget(self.mask_label)
        mask_layout.addWidget(self.masks_folder_input)
        mask_layout.addWidget(btn_browse_masks)
        config_layout.addLayout(mask_layout)
        
        # Parámetros
        params_layout = QHBoxLayout()
        
        self.patch_size_input = QSpinBox()
        self.patch_size_input.setRange(64, 1024)
        self.patch_size_input.setValue(128)
        params_layout.addWidget(QLabel("Patch Size:"))
        params_layout.addWidget(self.patch_size_input)
        
        self.stride_input = QSpinBox()
        self.stride_input.setRange(32, 512)
        self.stride_input.setValue(64)
        params_layout.addWidget(QLabel("Stride:"))
        params_layout.addWidget(self.stride_input)
        
        self.batch_size_input = QSpinBox()
        self.batch_size_input.setRange(1, 64)
        self.batch_size_input.setValue(4)
        params_layout.addWidget(QLabel("Batch Size:"))
        params_layout.addWidget(self.batch_size_input)
        
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 500)
        self.epochs_input.setValue(25)
        params_layout.addWidget(QLabel("Epochs:"))
        params_layout.addWidget(self.epochs_input)
        
        config_layout.addLayout(params_layout)
        
        # Backbone
        backbone_layout = QHBoxLayout()
        self.backbone_combo = QComboBox()
        self.backbone_combo.addItems(["resnet34", "resnet50", "efficientnetb0", "mobilenet"])
        backbone_layout.addWidget(QLabel("Backbone:"))
        backbone_layout.addWidget(self.backbone_combo)
        config_layout.addLayout(backbone_layout)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # === ACCIONES ===
        actions_layout = QHBoxLayout()
        self.btn_start = QPushButton("🚀 INICIAR ENTRENAMIENTO")
        self.btn_start.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-size: 14px; padding: 10px; }")
        self.btn_start.clicked.connect(self.start_training)
        
        self.btn_stop = QPushButton("⏹️ DETENER")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_training)
        
        actions_layout.addWidget(self.btn_start)
        actions_layout.addWidget(self.btn_stop)
        layout.addLayout(actions_layout)
        
        # === PROGRESO ===
        progress_group = QGroupBox("📊 Progreso")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Esperando...")
        self.metrics_label = QLabel("")
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.metrics_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # === LOGS ===
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        layout.addWidget(QLabel("📝 Logs:"))
        layout.addWidget(self.log_text)
    
    def browse_folder(self, target_input):
        """Abre diálogo para seleccionar carpeta"""
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta")
        if folder:
            target_input.setText(folder)
    
    def on_training_type_changed(self, index):
        """Maneja el cambio de tipo de entrenamiento."""
        is_supervised = (index == 0)
        
        # Habilitar/deshabilitar campos de máscaras
        self.mask_label.setEnabled(is_supervised)
        self.masks_folder_input.setEnabled(is_supervised)
        
        # Cambiar placeholder según el tipo
        if is_supervised:
            self.masks_folder_input.setPlaceholderText("Carpeta con máscaras .tif")
        else:
            self.masks_folder_input.setPlaceholderText("No requerido para entrenamiento no supervisado")
            self.masks_folder_input.clear()
    
    def start_training(self):
        """Inicia el entrenamiento"""
        images_folder = self.images_folder_input.text()
        masks_folder = self.masks_folder_input.text()
        is_supervised = (self.training_type_combo.currentIndex() == 0)
        
        # Validaciones
        if not images_folder:
            QMessageBox.warning(self, "Error", "Selecciona la carpeta de imágenes")
            return
        
        if not os.path.exists(images_folder):
            QMessageBox.warning(self, "Error", f"Carpeta de imágenes no existe:\n{images_folder}")
            return
        
        # Solo validar máscaras si es supervisado
        if is_supervised:
            if not masks_folder:
                QMessageBox.warning(self, "Error", "Selecciona la carpeta de máscaras (modo supervisado)")
                return
            
            if not os.path.exists(masks_folder):
                QMessageBox.warning(self, "Error", f"Carpeta de máscaras no existe:\n{masks_folder}")
                return
        
        try:
            if is_supervised:
                # Entrenamiento SUPERVISADO (U-Net con máscaras)
                response = self.api_client.start_training(
                    images_folder=images_folder,
                    masks_folder=masks_folder,
                    patch_size=self.patch_size_input.value(),
                    stride=self.stride_input.value(),
                    batch_size=self.batch_size_input.value(),
                    epochs=self.epochs_input.value(),
                    backbone=self.backbone_combo.currentText(),
                )
                self.log_text.append(f"✅ Entrenamiento SUPERVISADO iniciado: {response['job_id']}")
            else:
                # Entrenamiento NO SUPERVISADO (Autoencoder sin máscaras)
                response = self.api_client.start_unsupervised_training(
                    images_folder=images_folder,
                    batch_size=self.batch_size_input.value(),
                    epochs=self.epochs_input.value(),
                )
                self.log_text.append(f"✅ Entrenamiento NO SUPERVISADO iniciado: {response['job_id']}")
            
            self.current_job_id = response["job_id"]
            
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.timer.start(2000)  # Actualizar cada 2 segundos
            
            self.training_started.emit(self.current_job_id)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo iniciar el entrenamiento:\n{str(e)}")
            self.log_text.append(f"❌ Error: {str(e)}")
    
    def stop_training(self):
        """Detiene el entrenamiento"""
        if not self.current_job_id:
            return
        
        try:
            self.api_client.cancel_training(self.current_job_id)
            self.log_text.append(f"⏹️ Entrenamiento cancelado: {self.current_job_id}")
            self.reset_ui()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"No se pudo cancelar:\n{str(e)}")
    
    def update_progress(self):
        """Actualiza el progreso del entrenamiento"""
        if not self.current_job_id:
            return
        
        try:
            status = self.api_client.get_training_status(self.current_job_id)
            
            # Actualizar barra de progreso
            progress = int(status["progress"] * 100)
            self.progress_bar.setValue(progress)
            
            # Actualizar etiquetas
            epoch_text = f"Epoch {status['current_epoch']}/{status['total_epochs']} - {status['status']}"
            self.progress_label.setText(epoch_text)
            
            if status.get("metrics"):
                metrics = status["metrics"]
                metrics_text = (
                    f"Loss: {metrics.get('loss', 0):.4f} | "
                    f"IoU: {metrics.get('iou_score', 0):.4f} | "
                    f"Val Loss: {metrics.get('val_loss', 0):.4f} | "
                    f"Val IoU: {metrics.get('val_iou_score', 0):.4f}"
                )
                self.metrics_label.setText(metrics_text)
            
            # Si terminó
            if status["status"] in ["completed", "failed", "cancelled"]:
                self.timer.stop()
                self.reset_ui()
                
                if status["status"] == "completed":
                    self.log_text.append(f"✅ Entrenamiento completado exitosamente!")
                    QMessageBox.information(self, "Completado", "Entrenamiento finalizado correctamente")
                elif status["status"] == "failed":
                    self.log_text.append(f"❌ Error: {status.get('error_message', 'Desconocido')}")
                    QMessageBox.critical(self, "Error", f"Entrenamiento fallido:\n{status.get('error_message')}")
                
        except Exception as e:
            self.log_text.append(f"⚠️ Error al actualizar progreso: {str(e)}")
    
    def reset_ui(self):
        """Reinicia el estado de la UI"""
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.current_job_id = None
