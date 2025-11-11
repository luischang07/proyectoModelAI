"""Cliente API para comunicación con el backend"""
import requests
from typing import Optional, Dict, Any, List


class APIClient:
    """Cliente para interactuar con la API REST del backend"""
    
    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        self.base_url = base_url
        self.session = requests.Session()
    
    # === TRAINING ===
    def start_training(
        self,
        images_folder: str,
        masks_folder: str,
        patch_size: int = 256,
        stride: int = 128,
        batch_size: int = 8,
        epochs: int = 50,
        val_split: float = 0.2,
        backbone: str = "resnet34",
        encoder_weights: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Inicia un entrenamiento"""
        payload = {
            "images_folder": images_folder,
            "masks_folder": masks_folder,
            "patch_size": patch_size,
            "stride": stride,
            "batch_size": batch_size,
            "epochs": epochs,
            "val_split": val_split,
            "backbone": backbone,
            "encoder_weights": encoder_weights,
        }
        response = self.session.post(f"{self.base_url}/training/start", json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_training_status(self, job_id: str, is_unsupervised: bool = False) -> Dict[str, Any]:
        """Obtiene el estado de un entrenamiento (supervisado o no supervisado)"""
        endpoint = "unsupervised" if is_unsupervised else "training"
        response = self.session.get(f"{self.base_url}/{endpoint}/status/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def cancel_training(self, job_id: str, is_unsupervised: bool = False) -> Dict[str, Any]:
        """Cancela un entrenamiento (supervisado o no supervisado)"""
        endpoint = "unsupervised" if is_unsupervised else "training"
        response = self.session.delete(f"{self.base_url}/{endpoint}/cancel/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def start_unsupervised_training(
        self,
        images_folder: str,
        batch_size: int = 16,
        epochs: int = 50,
        latent_dim: int = 128,
        validation_split: float = 0.2,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Inicia un entrenamiento NO SUPERVISADO (Autoencoder).
        Solo requiere imágenes, NO necesita máscaras.
        """
        if model_name is None:
            from datetime import datetime
            model_name = f"autoencoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        payload = {
            "model_name": model_name,
            "images_folder": images_folder,
            "epochs": epochs,
            "batch_size": batch_size,
            "latent_dim": latent_dim,
            "validation_split": validation_split,
        }
        response = self.session.post(f"{self.base_url}/unsupervised/train", json=payload)
        response.raise_for_status()
        return response.json()
    
    # === INFERENCE ===
    def start_prediction(
        self,
        image_path: str,
        model_id: str,
        threshold: float = 0.5,
        stride: int = 256,
        batch_size: int = 16,
    ) -> Dict[str, Any]:
        """Inicia una predicción"""
        payload = {
            "image_path": image_path,
            "model_id": model_id,
            "threshold": threshold,
            "stride": stride,
            "batch_size": batch_size,
        }
        response = self.session.post(f"{self.base_url}/inference/predict", json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_inference_status(self, job_id: str) -> Dict[str, Any]:
        """Obtiene el estado de una inferencia"""
        response = self.session.get(f"{self.base_url}/inference/status/{job_id}")
        response.raise_for_status()
        return response.json()
    
    # === MODELS ===
    def list_models(self) -> List[Dict[str, Any]]:
        """Lista todos los modelos"""
        response = self.session.get(f"{self.base_url}/models/")
        response.raise_for_status()
        data = response.json()
        return data.get("models", [])
    
    def get_model(self, model_id: str) -> Dict[str, Any]:
        """Obtiene info de un modelo"""
        response = self.session.get(f"{self.base_url}/models/{model_id}")
        response.raise_for_status()
        return response.json()
    
    def delete_model(self, model_id: str) -> Dict[str, Any]:
        """Elimina un modelo"""
        response = self.session.delete(f"{self.base_url}/models/{model_id}")
        response.raise_for_status()
        return response.json()
    
    # === HEALTH ===
    def health_check(self) -> bool:
        """Verifica si el backend está disponible"""
        try:
            response = self.session.get(f"{self.base_url.replace('/api/v1', '')}/health")
            return response.status_code == 200
        except Exception:
            return False
