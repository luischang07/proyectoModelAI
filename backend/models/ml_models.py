"""Wrapper ML para modelos U-Net"""
import os
from datetime import datetime
from typing import Optional, Tuple
import numpy as np
from pathlib import Path

from backend.models.architecture_unet import build_unet
from backend.models.procesamiento import prepare_training_patches
from backend.config import settings


class MLModelWrapper:
    """Wrapper para facilitar operaciones con modelos U-Net"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str, compile_model: bool = False):
        """Carga un modelo guardado"""
        import importlib
        keras = importlib.import_module("tensorflow.keras")
        self.model = keras.models.load_model(model_path, compile=compile_model)
        self.model_path = model_path
        return self.model
    
    def build_and_compile(
        self,
        input_shape: Tuple[int, int, int],
        n_classes: int = 1,
        backbone: str = "resnet34",
        encoder_weights: Optional[str] = None,
    ):
        """Construye y compila un nuevo modelo"""
        self.model = build_unet(
            input_shape=input_shape,
            n_classes=n_classes,
            backbone=backbone,
            encoder_weights=encoder_weights
        )
        return self.model
    
    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 8,
        callbacks: Optional[list] = None,
    ):
        """Entrena el modelo"""
        if self.model is None:
            raise ValueError("Modelo no inicializado. Llama a build_and_compile() primero.")
        
        history = self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks or []
        )
        return history
    
    def predict(
        self,
        x: np.ndarray,
        batch_size: int = 16,
        threshold: float = 0.5
    ) -> np.ndarray:
        """Predice máscaras para un conjunto de imágenes"""
        if self.model is None:
            raise ValueError("Modelo no cargado. Llama a load_model() primero.")
        
        probs = self.model.predict(x, batch_size=batch_size, verbose=0)
        masks = (probs >= threshold).astype(np.uint8)
        return masks
    
    def save_model(self, save_dir: Optional[Path] = None) -> str:
        """Guarda el modelo"""
        if self.model is None:
            raise ValueError("No hay modelo para guardar")
        
        save_dir = save_dir or settings.MODELS_DIR
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = save_dir / f"unet_model_{timestamp}.keras"
        
        self.model.save(str(model_path))
        self.model_path = str(model_path)
        return str(model_path)
    
    def get_model_info(self) -> dict:
        """Obtiene información del modelo"""
        if self.model is None:
            return {}
        
        return {
            "input_shape": self.model.input_shape[1:],  # Sin batch dimension
            "output_shape": self.model.output_shape[1:],
            "num_parameters": self.model.count_params(),
            "num_layers": len(self.model.layers),
        }


def prepare_data_for_training(
    images_folder: str,
    masks_folder: str,
    patch_size: int = 256,
    stride: int = 128
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Prepara datos de múltiples imágenes para entrenamiento
    
    Returns:
        x_data, y_data, num_channels
    """
    from pathlib import Path
    import glob
    
    images_folder_path = Path(images_folder)
    masks_folder_path = Path(masks_folder)
    
    # Buscar archivos .tif/.tiff
    image_files = sorted(glob.glob(str(images_folder_path / "*.tif*")))
    mask_files = sorted(glob.glob(str(masks_folder_path / "*.tif*")))
    
    if len(image_files) == 0:
        raise ValueError(f"No se encontraron imágenes en {images_folder}")
    if len(mask_files) == 0:
        raise ValueError(f"No se encontraron máscaras en {masks_folder}")
    
    x_all, y_all = [], []
    num_channels = 0
    
    for img_path, mask_path in zip(image_files, mask_files):
        x, y, nc = prepare_training_patches(
            img_path,
            mask_path,
            patch_size=patch_size,
            stride=stride
        )
        if x.size > 0:
            x_all.append(x)
            y_all.append(y)
            num_channels = nc
    
    if not x_all:
        raise ValueError("No se generaron parches de ninguna imagen")
    
    x_data = np.concatenate(x_all, axis=0)
    y_data = np.concatenate(y_all, axis=0)
    
    return x_data, y_data, num_channels
