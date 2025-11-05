"""
Script para generar máscaras en cero (todo negro) a partir de imágenes .tif

Este script crea máscaras binarias completamente en 0 (negro) para imágenes de cultivo sano.
Útil cuando tienes solo imágenes normales y quieres entrenar el modelo para aprender
cómo se ve el cultivo sano.

Uso:
    python generate_zero_masks.py
"""

import os
from pathlib import Path
import rasterio
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    # Fallback si no está instalado tqdm
    def tqdm(iterable, desc=""):
        return iterable


def generate_zero_masks(
    images_folder: str,
    masks_folder: str,
    overwrite: bool = False
):
    """
    Genera máscaras en cero para todas las imágenes .tif en la carpeta especificada.
    
    Args:
        images_folder: Ruta a la carpeta con imágenes originales
        masks_folder: Ruta donde se guardarán las máscaras
        overwrite: Si True, sobrescribe máscaras existentes
    """
    
    images_path = Path(images_folder)
    masks_path = Path(masks_folder)
    
    # Verificar que existe la carpeta de imágenes
    if not images_path.exists():
        print(f"❌ Error: La carpeta {images_folder} no existe")
        return
    
    # Crear carpeta de máscaras si no existe
    masks_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Carpeta de máscaras: {masks_path}")
    
    # Obtener todas las imágenes .tif
    image_files = list(images_path.glob("*.tif")) + list(images_path.glob("*.tiff"))
    
    if not image_files:
        print(f"❌ No se encontraron archivos .tif en {images_folder}")
        return
    
    print(f"\n📊 Encontradas {len(image_files)} imágenes")
    print("🔄 Generando máscaras en cero...\n")
    
    created = 0
    skipped = 0
    errors = 0
    
    for image_file in tqdm(image_files, desc="Procesando"):
        try:
            # Nombre de la máscara (mismo nombre que la imagen)
            mask_file = masks_path / image_file.name
            
            # Verificar si ya existe
            if mask_file.exists() and not overwrite:
                skipped += 1
                continue
            
            # Leer la imagen para obtener dimensiones y perfil
            with rasterio.open(image_file) as src:
                # Obtener dimensiones
                height = src.height
                width = src.width
                
                # Copiar perfil (metadatos geoespaciales)
                profile = src.profile.copy()
                
                # Actualizar perfil para máscara binaria
                profile.update({
                    'count': 1,           # Una sola banda
                    'dtype': 'uint8',     # Tipo de dato binario
                    'compress': 'lzw'     # Compresión para ahorrar espacio
                })
            
            # Crear máscara en cero (todo negro)
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Guardar máscara
            with rasterio.open(mask_file, 'w', **profile) as dst:
                dst.write(mask, 1)
            
            created += 1
            
        except Exception as e:
            print(f"\n❌ Error procesando {image_file.name}: {str(e)}")
            errors += 1
    
    # Resumen
    print(f"\n{'='*60}")
    print(f"✅ Máscaras creadas: {created}")
    if skipped > 0:
        print(f"⏭️  Máscaras omitidas (ya existían): {skipped}")
    if errors > 0:
        print(f"❌ Errores: {errors}")
    print(f"{'='*60}\n")
    
    if created > 0:
        print(f"📁 Las máscaras se guardaron en: {masks_path}")
        print(f"\n💡 Ahora puedes entrenar el modelo con:")
        print(f"   - Carpeta de imágenes: {images_path}")
        print(f"   - Carpeta de máscaras: {masks_path}")


def main():
    """Función principal con configuración por defecto"""
    
    print("=" * 60)
    print("🎯 Generador de Máscaras en Cero para Cultivo Sano")
    print("=" * 60)
    
    # Configuración por defecto
    default_images = "data/train"
    default_masks = "data/masks"
    
    # Pedir carpeta de imágenes
    print(f"\n📂 Carpeta de imágenes originales")
    images_input = input(f"   (Enter para usar '{default_images}'): ").strip()
    images_folder = images_input if images_input else default_images
    
    # Pedir carpeta de máscaras
    print(f"\n📂 Carpeta donde guardar las máscaras")
    masks_input = input(f"   (Enter para usar '{default_masks}'): ").strip()
    masks_folder = masks_input if masks_input else default_masks
    
    # Preguntar si sobrescribir
    print(f"\n⚠️  ¿Sobrescribir máscaras existentes?")
    overwrite_input = input("   (s/N): ").strip().lower()
    overwrite = overwrite_input in ['s', 'si', 'sí', 'y', 'yes']
    
    print()
    
    # Generar máscaras
    generate_zero_masks(images_folder, masks_folder, overwrite)
    
    print("✨ Proceso completado!\n")


if __name__ == "__main__":
    main()
