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
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    
    # Celery broker y backend (soporta REDIS_URL o construcción desde componentes)
    @property
    def celery_broker_url(self) -> str:
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            return redis_url
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    @property
    def celery_result_backend(self) -> str:
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            return redis_url
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
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
