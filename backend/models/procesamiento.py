import os
from typing import Tuple, List, Optional

import numpy as np
import rasterio
from sklearn.preprocessing import MinMaxScaler

# Tamaño por defecto de parches (reducido para ahorrar memoria)
PATCH_SIZE = 128

def load_image(image_path: str) -> Tuple[np.ndarray, dict]:
    """Carga una imagen multiespectral .tif como array (H, W, C) y retorna su perfil.

    Devuelve:
      - image: np.ndarray de shape (alto, ancho, canales)
      - profile: metadatos del raster (crs, transform, dtype, etc.)
    """
    with rasterio.open(image_path) as src:
        image = np.transpose(src.read(), (1, 2, 0))
        profile = src.profile.copy()
    return image, profile

def load_mask(mask_path: str) -> Tuple[np.ndarray, dict]:
    """Carga una máscara binaria .tif como array (H, W) y retorna su perfil."""
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
        np.stack(y_list, axis=0).astype(np.float32)  # Cambiar a float32 para TensorFlow
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

__all__ = [
    "PATCH_SIZE",
    "load_image",
    "load_mask",
    "normalize_image",
    "create_patches",
    "reconstruct_from_patches",
    "save_mask_geotiff",
    "prepare_training_patches",
]