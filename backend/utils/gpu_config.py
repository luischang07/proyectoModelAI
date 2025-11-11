"""
Configuración de GPU para PyTorch

Este módulo detecta y configura el uso de GPU automáticamente usando PyTorch.
"""

import os

from typing import List, Dict, Any


def get_gpu_info() -> Dict[str, Any]:
    """
    Obtiene información sobre las GPUs disponibles.
    
    Returns:
        dict: Información de GPUs con nombre, memoria, estado
    """
    # Intentar importar y listar GPUs varias veces para tolerar fallos transitorios
    # Use PyTorch to detect GPU availability. Retries help in case of transient
    # initialization races (e.g. WSL driver warmup).
    import time
    try:
        import torch
    except Exception as e:
        return {
            "gpu_available": False,
            "gpu_count": 0,
            "gpus": [],
            "framework_version": None,
            "cuda_available": False,
            "using_device": "CPU",
            "error": repr(e)
        }

    # Retry a few times if needed (transient cuInit failures)
    gpu_count = 0
    for attempt in range(1, 4):
        try:
            gpu_count = torch.cuda.device_count()
            break
        except Exception:
            # transient driver/initialization issue — wait and retry
            print(f"⚠️  torch.cuda.device_count attempt {attempt} failed, retrying")
            time.sleep(attempt)

    info = {
        "gpu_available": torch.cuda.is_available() and gpu_count > 0,
        "gpu_count": gpu_count,
        "gpus": [],
        "framework_version": getattr(torch, '__version__', None),
        "cuda_available": torch.cuda.is_available(),
        "using_device": "GPU" if (torch.cuda.is_available() and gpu_count > 0) else "CPU",
    }
    
    if info["gpu_count"]:
        for i in range(info["gpu_count"]):
            try:
                name = torch.cuda.get_device_name(i)
            except Exception:
                name = f"cuda:{i}"
            info["gpus"].append({"id": i, "name": name, "device_type": "GPU"})
    
    return info


def configure_gpu(memory_limit_mb: int = None, allow_growth: bool = True) -> bool:
    """
    Configura (de forma limitada) el uso de GPU para PyTorch.

    Args:
        memory_limit_mb: Límite de memoria en MB (None = sin límite). Nota: la
            limitación precisa puede no ser soportada en todas las versiones de
            PyTorch/driver.
        allow_growth: Ignorado en PyTorch (manejable por allocator).

    Returns:
        bool: True si se detectó/configuró GPU exitosamente
    """
    try:
        import torch
    except Exception as e:
        print(f"⚠️  PyTorch no disponible: {e}")
        return False

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        print("⚠️  No se detectaron GPUs (PyTorch). Usando CPU.")
        return False

    try:
        # Habilitar optimizaciones de cudnn
        torch.backends.cudnn.benchmark = True
        print(f"✅ PyTorch detectó {torch.cuda.device_count()} GPU(s)")

        # Intentar limitar memoria por proceso si se solicita (API no garantizada)
        if memory_limit_mb:
            try:
                total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                fraction = float(memory_limit_mb) / float(total)
                # set_per_process_memory_fraction está disponible en PyTorch >=1.10
                torch.cuda.set_per_process_memory_fraction(fraction, 0)
                print(f"✅ Limitado uso de memoria a ~{memory_limit_mb} MB (frac {fraction:.3f})")
            except Exception as e:
                print(f"⚠️  No fue posible limitar memoria por proceso: {e}")

        return True
    except Exception as e:
        print(f"❌ Error configurando PyTorch GPU: {e}")
        return False


def enable_mixed_precision() -> None:
    """
    Indica soporte para mixed precision en PyTorch. En PyTorch el uso de AMP se
    aplica en el bucle de entrenamiento (torch.cuda.amp.autocast + GradScaler).
    Aquí solo imprimimos información y habilitamos ciertas banderas.
    """
    try:
        import torch
        # Nothing to set globally; training loop should use autocast/GradScaler.
        print("ℹ️  Habilite torch.cuda.amp.autocast y torch.cuda.amp.GradScaler en el bucle de entrenamiento para usar FP16.")
    except Exception as e:
        print(f"⚠️  No se pudo preparar mixed precision: {e}")


def set_deterministic_ops(seed: int = 42) -> None:
    """
    Configura comportamiento determinístico para PyTorch/NumPy/Python.
    """
    import random
    import numpy as np
    try:
        import torch
    except Exception:
        torch = None

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
            print(f"✅ PyTorch ops determinísticas habilitadas (seed={seed})")
        except Exception:
            print(f"⚠️  No fue posible habilitar todas las operaciones determinísticas en esta versión de PyTorch")


def print_device_info() -> None:
    """Imprime información detallada sobre los dispositivos disponibles."""
    print("\n" + "="*60)
    print("🖥️  INFORMACIÓN DE DISPOSITIVOS")
    print("="*60)
    
    info = get_gpu_info()
    
    print(f"Framework Version: {info.get('framework_version')}")
    print(f"CUDA Disponible: {'✅ Sí' if info['cuda_available'] else '❌ No'}")
    print(f"Dispositivo en uso: {info['using_device']}")
    print(f"GPUs detectadas: {info['gpu_count']}")
    
    if info['gpus']:
        print("\n📊 Detalles de GPUs:")
        for gpu in info['gpus']:
            print(f"  • GPU {gpu['id']}: {gpu['name']}")
    else:
        print("\n⚠️  No se detectaron GPUs. El entrenamiento usará CPU.")
        print("   💡 Para acelerar, instala CUDA + PyTorch con soporte GPU")
    
    # Información adicional de memoria (PyTorch)
    try:
        import torch
        if torch.cuda.is_available():
            print("\n💾 Configuración de Memoria GPU (estimada):")
            for i in range(torch.cuda.device_count()):
                try:
                    props = torch.cuda.get_device_properties(i)
                    total = props.total_memory / (1024**3)
                    print(f"  • cuda:{i} ({props.name}): {total:.2f} GB total")
                except Exception:
                    print(f"  • cuda:{i}: info no disponible")
    except Exception:
        pass
    
    print("="*60 + "\n")


def auto_configure(
    enable_mixed_precision_training: bool = False,
    memory_limit_mb: int = None,
    allow_growth: bool = True,
    deterministic: bool = False,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Configuración automática de GPU con opciones comunes.
    
    Args:
        enable_mixed_precision_training: Habilitar FP16 (más rápido en GPUs modernas)
        memory_limit_mb: Límite de memoria GPU en MB (None = sin límite)
        allow_growth: Permitir crecimiento dinámico de memoria
        deterministic: Habilitar operaciones determinísticas
        seed: Semilla para reproducibilidad
        verbose: Imprimir información detallada
    
    Returns:
        dict: Información de configuración
    """
    if verbose:
        print_device_info()
    
    # Configurar GPU
    gpu_configured = configure_gpu(memory_limit_mb=memory_limit_mb, allow_growth=allow_growth)
    
    # Precisión mixta si se solicita
    if enable_mixed_precision_training and gpu_configured:
        enable_mixed_precision()
    
    # Reproducibilidad si se solicita
    if deterministic:
        set_deterministic_ops(seed=seed)
    
    return get_gpu_info()


# NO auto-configurar al importar - debe hacerse explícitamente dentro de las tareas
# para evitar problemas con Celery fork y CUDA initialization

__all__ = [
    "get_gpu_info",
    "configure_gpu",
    "enable_mixed_precision",
    "set_deterministic_ops",
    "print_device_info",
    "auto_configure"
]
