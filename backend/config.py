"""Configuración de la aplicación"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Directorio base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Configuración general de la aplicación"""
    
    # API
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "U-Net Anomaly Detection API"
    VERSION: str = "1.0.0"
    
    # CORS
    BACKEND_CORS_ORIGINS: list = ["http://localhost:8000", "http://127.0.0.1:8000"]
    
    # Database
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/app.db"
    
    # Redis (para Celery)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    CELERY_BROKER_URL: str = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    CELERY_RESULT_BACKEND: str = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    
    # Directorios
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    OUTPUT_DIR: Path = BASE_DIR / "output"
    LOGS_DIR: Path = BASE_DIR / "logs"
    
    # ML Config
    DEFAULT_PATCH_SIZE: int = 256
    DEFAULT_STRIDE: int = 128
    DEFAULT_BATCH_SIZE: int = 8
    DEFAULT_EPOCHS: int = 50
    DEFAULT_BACKBONE: str = "resnet34"
    
    # Limits
    MAX_UPLOAD_SIZE: int = 5 * 1024 * 1024 * 1024  # 5 GB
    
    class Config:
        case_sensitive = True
        env_file = ".env"


# Instancia global
settings = Settings()

# Crear directorios si no existen
for directory in [settings.DATA_DIR, settings.MODELS_DIR, settings.OUTPUT_DIR, settings.LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
