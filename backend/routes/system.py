"""
Rutas para información del sistema y GPU
"""

from fastapi import APIRouter
from typing import Dict, Any

from backend.utils.gpu_config import get_gpu_info

router = APIRouter(prefix="/system", tags=["System"])


@router.get("/gpu", response_model=Dict[str, Any])
def get_gpu_status():
    """
    Obtiene información sobre las GPUs disponibles.
    
    Returns:
        dict: Información de GPUs, CUDA, y dispositivo en uso
    """
    return get_gpu_info()


@router.get("/info", response_model=Dict[str, Any])
def get_system_info():
    """
    Obtiene información general del sistema.
    
    Returns:
        dict: Información del sistema
    """
    import platform
    import sys
    
    gpu_info = get_gpu_info()
    
    return {
        "system": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "architecture": platform.machine()
        },
        "gpu": gpu_info
    }
