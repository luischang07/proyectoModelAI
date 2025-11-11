import os
import platform
from typing import Tuple, List, Optional
from pathlib import Path

import numpy as np
import rasterio
from sklearn.preprocessing import MinMaxScaler

# Tamaño por defecto de parches (reducido para ahorrar memoria)
PATCH_SIZE = 128

def convert_windows_path_to_wsl(path: str) -> str:
    """
    Convierte una ruta de Windows a formato WSL2 o Docker según el entorno.
    
    Args:
        path: Ruta que puede ser de Windows (C:/...) o ya estar en formato WSL (/mnt/c/...)
        
    Returns:
        Ruta convertida al formato apropiado
    """
    # Si estamos en Windows, no hacer conversión
    if platform.system() == "Windows":
        return path
    
    # Detectar si estamos en Docker (verificar si existe /.dockerenv)
    in_docker = os.path.exists("/.dockerenv")
    
    # Si ya está en formato WSL, retornar tal cual (solo si NO estamos en Docker)
    if path.startswith("/mnt/") and not in_docker:
        return path
    
    # Si estamos en Docker y recibimos una ruta absoluta de Windows o WSL
    if in_docker and (":" in path or path.startswith("/mnt/")):
        # Extraer la parte relevante de la ruta
        # Ejemplo: C:/Users/.../proyectoModelAI/data/train -> data/train
        # O: /mnt/c/Users/.../proyectoModelAI/data/train -> data/train
        
        # Convertir \ a / primero
        path = path.replace("\\", "/")
        
        # Buscar patrones comunes de carpetas del proyecto
        for folder in ["data", "models", "output", "logs"]:
            if f"/{folder}/" in path or path.endswith(f"/{folder}"):
                # Encontrar la posición de la carpeta
                idx = path.rfind(f"/{folder}/")
                if idx != -1:
                    # Retornar la ruta relativa desde /app en Docker
                    relative = path[idx + 1:]  # Quitar el / inicial
                    return f"/app/{relative}"
                elif path.endswith(f"/{folder}"):
                    return f"/app/{folder}"
        
        # Si no encontramos un patrón conocido, intentar usar la ruta tal cual
        print(f"⚠️  Advertencia: No se pudo convertir la ruta a formato Docker: {path}")
        return path
    
    # Convertir C:/ a /mnt/c/ (solo para WSL, no Docker)
    if ":" in path and not in_docker:
        # Extraer letra de unidad
        drive_letter = path[0].lower()
        # Remover C:/ o C:\ y convertir \ a /
        rest_of_path = path[3:].replace("\\", "/")  # path[3:] para saltar "C:/" o "C:\"
        wsl_path = f"/mnt/{drive_letter}/{rest_of_path}"
        return wsl_path
    
    # Path no necesita conversión
    return path

def load_image(image_path: str) -> Tuple[np.ndarray, dict]:
    """Carga una imagen multiespectral .tif como array (H, W, C) y retorna su perfil.

    Devuelve:
      - image: np.ndarray de shape (alto, ancho, canales)
      - profile: metadatos del raster (crs, transform, dtype, etc.)
    """
    # Convertir ruta de Windows a WSL si es necesario
    image_path = convert_windows_path_to_wsl(image_path)
    
    with rasterio.open(image_path) as src:
        image = np.transpose(src.read(), (1, 2, 0))
        profile = src.profile.copy()
    return image, profile

def load_mask(mask_path: str) -> Tuple[np.ndarray, dict]:
    """Carga una máscara binaria .tif como array (H, W) y retorna su perfil."""
    # Convertir ruta de Windows a WSL si es necesario
    mask_path = convert_windows_path_to_wsl(mask_path)
    
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        profile = src.profile.copy()
    # Forzar binaria (0/1)
    mask = (mask > 0).astype(np.uint8)
    return mask, profile

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normaliza cada canal de la imagen a [0, 1] usando MinMax por canal.

    Nota: Normaliza por imagen (no guarda el scaler). Para inferencia por imagen
    suele ser suficiente; si se requiere consistencia global, persistir min/max.
    """
    h, w, c = image.shape
    image_2d = image.reshape(-1, c)
    scaler = MinMaxScaler()
    image_norm = scaler.fit_transform(image_2d)
    # Convertir a float32 para reducir uso de memoria a la mitad
    return image_norm.reshape(h, w, c).astype(np.float32)

def create_patches(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    patch_size: int = PATCH_SIZE,
    stride: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[Tuple[int, int]]]:
    """Corta la imagen (y opcionalmente la máscara) en parches.

    Parámetros:
      - image: (H, W, C)
      - mask:  (H, W) o None si no hay máscara
      - patch_size: tamaño del parche cuadrado
      - stride: paso entre parches (por defecto = patch_size, sin traslape)

    Devuelve:
      - X_patches: (N, patch, patch, C)
      - y_patches: (N, patch, patch, 1) o None
      - coords: lista de (y, x) superiores izquierdas para reconstrucción
    """
    if stride is None:
        stride = patch_size

    H, W, C = image.shape
    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    coords: List[Tuple[int, int]] = []

    y_has_mask = mask is not None

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            x_list.append(image[y:y + patch_size, x:x + patch_size])
            coords.append((y, x))
            if y_has_mask:
                y_patch = mask[y:y + patch_size, x:x + patch_size]
                y_list.append(y_patch[..., None])  # (p,p,1)

    x_patches = np.stack(x_list, axis=0) if x_list else np.empty((0, patch_size, patch_size, C))
    y_patches = (
        np.stack(y_list, axis=0).astype(np.float32)
        if y_has_mask and y_list
        else None
    )
    return x_patches, y_patches, coords

def reconstruct_from_patches(
    patches: np.ndarray,
    coords: List[Tuple[int, int]],
    out_shape: Tuple[int, int],
    patch_size: int = PATCH_SIZE,
    reduce: str = "mean",
) -> np.ndarray:
    """Reconstruye un mosaico 2D a partir de parches y coordenadas.

    - patches: (N, patch, patch) o (N, patch, patch, 1)
    - coords: lista de (y, x)
    - out_shape: (H, W)
    - reduce: forma de combinar traslapes: 'mean' o 'max'
    """
    if patches.ndim == 4 and patches.shape[-1] == 1:
        patches = patches[..., 0]

    H, W = out_shape
    if reduce == "mean":
        acc = np.zeros((H, W), dtype=np.float32)
        count = np.zeros((H, W), dtype=np.float32)
        for (py, px), patch in zip(coords, patches):
            acc[py:py + patch_size, px:px + patch_size] += patch.astype(np.float32)
            count[py:py + patch_size, px:px + patch_size] += 1.0
        count[count == 0] = 1.0
        return acc / count
    elif reduce == "max":
        out = np.full((H, W), -np.inf, dtype=np.float32)
        for (py, px), patch in zip(coords, patches):
            region = out[py:py + patch_size, px:px + patch_size]
            np.maximum(region, patch.astype(np.float32), out=region)
        out[~np.isfinite(out)] = 0.0
        return out
    else:
        raise ValueError("reduce debe ser 'mean' o 'max'")

def save_mask_geotiff(
    output_path: str,
    mask: np.ndarray,
    reference_profile: dict,
    dtype: str = "uint8",
) -> None:
    """Guarda una máscara 2D como GeoTIFF usando el perfil de referencia."""
    profile = reference_profile.copy()
    profile.update({
        "count": 1,
        "dtype": dtype,
    })
    # Asegurar dimensiones correctas
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask.astype(dtype), 1)

def prepare_training_patches(
    image_path: str,
    mask_path: str,
    patch_size: int = PATCH_SIZE,
    stride: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Convenience: carga, normaliza, parchea y devuelve X, y y número de canales."""
    image, _ = load_image(image_path)
    mask, _ = load_mask(mask_path)
    image = normalize_image(image)
    X, y, _ = create_patches(image, mask, patch_size=patch_size, stride=stride)
    return X, y, X.shape[-1] if X.size else 0

def load_images_only(images_folder: str, max_images: int = None, cancel_check_fn=None) -> np.ndarray:
    """
    Carga SOLO imágenes (sin máscaras) para entrenamiento NO SUPERVISADO.
    
    Args:
        images_folder: Ruta a carpeta con imágenes
        max_images: Número máximo de imágenes a cargar (None = todas)
        cancel_check_fn: Función opcional que retorna True si se debe cancelar
    
    Devuelve:
        np.ndarray: Array con todos los parches de todas las imágenes (N, H, W, C)
    """
    from pathlib import Path
    
    # Convertir ruta de Windows a WSL si es necesario
    images_folder = convert_windows_path_to_wsl(images_folder)
    
    images_path = Path(images_folder)
    image_files = sorted(images_path.glob("*.tif"))
    
    if not image_files:
        raise ValueError(f"No se encontraron archivos .tif en {images_folder}")
    
    # Limitar número de imágenes solo si se especifica
    if max_images is not None and len(image_files) > max_images:
        print(f"⚠️  Encontradas {len(image_files)} imágenes, limitando a {max_images} primeras")
        image_files = image_files[:max_images]
    
    all_patches = []
    
    for i, img_file in enumerate(image_files):
        # Verificar cancelación antes de procesar cada imagen
        if cancel_check_fn and cancel_check_fn():
            raise RuntimeError("🛑 Carga de imágenes cancelada por el usuario")
        
        print(f"📷 Cargando imagen {i+1}/{len(image_files)}: {img_file.name}")
        
        # Cargar y normalizar imagen
        image, _ = load_image(str(img_file))
        image = normalize_image(image)
        
        # Crear parches (sin máscara)
        patches, _, _ = create_patches(image, mask=None, patch_size=PATCH_SIZE)
        
        if patches.size > 0:
            all_patches.append(patches)
            print(f"  ✅ {patches.shape[0]} parches generados")
    
    if not all_patches:
        raise ValueError("No se pudieron generar parches de las imágenes")
    
    total_patches = np.concatenate(all_patches, axis=0)
    print(f"\n📊 Total: {total_patches.shape[0]} parches de {len(image_files)} imágenes")
    return total_patches

__all__ = [
    "PATCH_SIZE",
    "load_image",
    "load_mask",
    "normalize_image",
    "create_patches",
    "reconstruct_from_patches",
    "save_mask_geotiff",
    "prepare_training_patches",
    "load_images_only",
    "LazyImageDataset",
    "get_image_paths",
]


def get_image_paths(images_folder: str) -> List[str]:
    """
    Obtiene todas las rutas de imágenes .tif en una carpeta.
    
    Args:
        images_folder: Carpeta con imágenes
        
    Returns:
        Lista de rutas absolutas a archivos .tif
    """
    images_folder = convert_windows_path_to_wsl(images_folder)
    folder_path = Path(images_folder)
    
    if not folder_path.exists():
        raise FileNotFoundError(f"❌ Carpeta no encontrada: {images_folder}")
    
    image_paths = sorted([
        str(p) for p in folder_path.glob("*.tif")
    ])
    
    print(f"📂 Encontradas {len(image_paths)} imágenes en {images_folder}")
    return image_paths


class LazyImageDataset:
    """
    Dataset de PyTorch que carga imágenes bajo demanda (lazy loading).
    Esto ahorra RAM al no cargar todas las imágenes en memoria.
    """
    def __init__(self, image_paths: List[str], patch_size: int = PATCH_SIZE, 
                 transform=None, cancel_check_fn=None):
        """
        Args:
            image_paths: Lista de rutas a archivos .tif
            patch_size: Tamaño de parches a usar
            transform: Transformaciones opcionales
            cancel_check_fn: Función para verificar cancelación
        """
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.transform = transform
        self.cancel_check_fn = cancel_check_fn
        
        # Cachear información básica (dimensiones) sin cargar imágenes
        self._cache_image_info()
    
    def _cache_image_info(self):
        """Cachear solo información básica de las imágenes."""
        if len(self.image_paths) == 0:
            raise ValueError("❌ No se proporcionaron rutas de imágenes")
        
        # Cargar solo la primera imagen para obtener dimensiones
        first_img, _ = load_image(self.image_paths[0])
        self.num_channels = first_img.shape[-1]
        
        print(f"📊 Dataset: {len(self.image_paths)} imágenes, {self.num_channels} canales")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Carga una imagen solo cuando se solicita."""
        import torch
        import torch.nn.functional as F
        
        # Verificar cancelación si está disponible
        if self.cancel_check_fn and self.cancel_check_fn():
            raise RuntimeError("🛑 Entrenamiento cancelado por el usuario")
        
        # Cargar imagen
        img_path = self.image_paths[idx]
        img, _ = load_image(img_path)
        
        # Convertir a tensor PyTorch (H,W,C) -> (C,H,W)
        img_tensor = torch.from_numpy(img.astype('float32')).permute(2, 0, 1)
        
        # Normalizar a rango [0, 1]
        img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-8)
        
        # Resize a patch_size si es necesario usando PyTorch
        _, H, W = img_tensor.shape
        if H != self.patch_size or W != self.patch_size:
            # Agregar batch dimension, resize, y remover batch dimension
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0), 
                size=(self.patch_size, self.patch_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Aplicar transformaciones si existen
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor