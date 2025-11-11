"""Wrapper ML para modelos U-Net y Autoencoder en PyTorch"""
import os
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from backend.models.architecture_unet import build_unet
from backend.models.procesamiento import prepare_training_patches, PATCH_SIZE
from backend.config import settings


class ConvAutoencoder(nn.Module):
    """Autoencoder convolucional para detección de anomalías (no supervisado)"""
    def __init__(self, in_channels=3, latent_dim=128, img_size=128):
        super().__init__()
        # Calcular tamaño después de 3 convoluciones con stride=2
        encoded_size = img_size // 8
        flattened_size = 128 * encoded_size * encoded_size
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(flattened_size, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, flattened_size),
            nn.Unflatten(1, (128, encoded_size, encoded_size)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, in_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        xrec = self.decoder(z)
        return xrec


class MLModelWrapper:
    """Wrapper para facilitar operaciones con modelos U-Net en PyTorch"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = None
        self.criterion = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str, compile_model: bool = False):
        """Carga un modelo guardado (supervisado o no supervisado)"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Verificar si es el formato nuevo (diccionario) o antiguo (solo state_dict)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Formato nuevo con metadata
            model_config = checkpoint.get('model_config', {})
            state_dict = checkpoint['model_state_dict']
            model_type = model_config.get('model_type', 'supervised_unet')
        else:
            # Formato antiguo (solo state_dict guardado directamente)
            state_dict = checkpoint
            model_config = {}
            
            # Detectar tipo por las claves del state_dict
            first_key = next(iter(state_dict.keys()))
            if 'encoder.0.weight' in state_dict or first_key.startswith('encoder'):
                model_type = 'unsupervised_autoencoder'
            else:
                model_type = 'supervised_unet'
        
        # Reconstruir el modelo según su tipo
        if model_type == 'unsupervised_autoencoder':
            # Modelo no supervisado (Autoencoder)
            self.model = ConvAutoencoder(
                in_channels=model_config.get('in_channels', 3),
                latent_dim=model_config.get('latent_dim', 128),
                img_size=model_config.get('img_size', PATCH_SIZE)
            )
        else:
            # Modelo supervisado (UNet)
            self.model = build_unet(
                input_shape=model_config.get('input_shape', (3, 256, 256)),
                n_classes=model_config.get('n_classes', 1)
            )
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.model_path = model_path
        self.model_type = model_type
        return self.model
    
    def build_and_compile(
        self,
        input_shape: Tuple[int, int, int],
        n_classes: int = 1,
        backbone: str = "resnet34",
        encoder_weights: Optional[str] = None,
        learning_rate: float = 0.001,
    ):
        """Construye y compila un nuevo modelo"""
        self.model = build_unet(
            input_shape=input_shape,
            n_classes=n_classes,
            backbone=backbone,
            encoder_weights=encoder_weights
        )
        self.model.to(self.device)
        
        # Define loss y optimizer
        self.criterion = nn.BCELoss() if n_classes == 1 else nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
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
    ) -> Dict[str, Any]:
        """Entrena el modelo"""
        if self.model is None:
            raise ValueError("Modelo no inicializado. Llama a build_and_compile() primero.")
        
        # Convertir datos a tensores PyTorch
        # Asumir formato (N, H, W, C) y convertir a (N, C, H, W)
        if x_train.ndim == 4 and x_train.shape[-1] <= 16:  # Último dim es canales
            x_train = np.transpose(x_train, (0, 3, 1, 2))
            x_val = np.transpose(x_val, (0, 3, 1, 2))
            y_train = np.transpose(y_train, (0, 3, 1, 2)) if y_train.ndim == 4 else y_train[..., None]
            y_val = np.transpose(y_val, (0, 3, 1, 2)) if y_val.ndim == 4 else y_val[..., None]
        
        # Normalizar si es necesario
        if x_train.max() > 2.0:
            x_train = x_train / 255.0
            x_val = x_val / 255.0
        
        # Crear DataLoaders
        train_dataset = TensorDataset(
            torch.FloatTensor(x_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(x_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Tracking history
        history = {'loss': [], 'iou_score': [], 'val_loss': [], 'val_iou_score': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_iou = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                train_iou += self._compute_iou(outputs, batch_y)
            
            train_loss /= len(train_loader)
            train_iou /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_iou = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    val_iou += self._compute_iou(outputs, batch_y)
            
            val_loss /= len(val_loader)
            val_iou /= len(val_loader)
            
            # Store history
            history['loss'].append(train_loss)
            history['iou_score'].append(train_iou)
            history['val_loss'].append(val_loss)
            history['val_iou_score'].append(val_iou)
            
            # Call callbacks if provided
            if callbacks:
                logs = {
                    'loss': train_loss,
                    'iou_score': train_iou,
                    'val_loss': val_loss,
                    'val_iou_score': val_iou
                }
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, logs)
        
        # Create history object
        class History:
            def __init__(self, history_dict):
                self.history = history_dict
        
        return History(history)
    
    def _compute_iou(self, pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
        """Calcula IoU (Intersection over Union)"""
        pred_mask = (pred > threshold).float()
        target_mask = target.float()
        
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum() - intersection
        
        iou = (intersection + 1e-7) / (union + 1e-7)
        return iou.item()
    
    def predict(
        self,
        x: np.ndarray,
        batch_size: int = 16,
        threshold: float = 0.5
    ) -> np.ndarray:
        """Predice máscaras para un conjunto de imágenes"""
        if self.model is None:
            raise ValueError("Modelo no cargado. Llama a load_model() primero.")
        
        self.model.eval()
        
        # Convertir a tensor PyTorch y ajustar formato
        if x.ndim == 4 and x.shape[-1] <= 16:
            x = np.transpose(x, (0, 3, 1, 2))
        
        if x.max() > 2.0:
            x = x / 255.0
        
        x_tensor = torch.FloatTensor(x).to(self.device)
        
        # Predecir en batches
        predictions = []
        with torch.no_grad():
            for i in range(0, len(x_tensor), batch_size):
                batch = x_tensor[i:i+batch_size]
                pred = self.model(batch)
                predictions.append(pred.cpu().numpy())
        
        probs = np.concatenate(predictions, axis=0)
        masks = (probs >= threshold).astype(np.uint8)
        
        # Convertir de vuelta a formato (N, H, W, C)
        if masks.ndim == 4:
            masks = np.transpose(masks, (0, 2, 3, 1))
        
        return masks
    
    def save_model(self, save_dir: Optional[Path] = None) -> str:
        """Guarda el modelo"""
        if self.model is None:
            raise ValueError("No hay modelo para guardar")
        
        save_dir = save_dir or settings.MODELS_DIR
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = save_dir / f"unet_model_{timestamp}.pt"
        
        # Guardar modelo y configuración
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_shape': (self.model.in_channels, 256, 256),
                'n_classes': self.model.n_classes
            }
        }, str(model_path))
        
        self.model_path = str(model_path)
        return str(model_path)
    
    def get_model_info(self) -> dict:
        """Obtiene información del modelo"""
        if self.model is None:
            return {}
        
        # Contar parámetros
        num_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            "input_shape": (self.model.in_channels, 256, 256),
            "output_shape": (self.model.n_classes, 256, 256),
            "num_parameters": num_params,
            "num_layers": len(list(self.model.modules())),
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
    from backend.models.procesamiento import convert_windows_path_to_wsl
    
    # Convertir rutas de Windows a WSL si es necesario
    images_folder = convert_windows_path_to_wsl(images_folder)
    masks_folder = convert_windows_path_to_wsl(masks_folder)
    
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
