# 🚀 Optimizaciones de Memoria para Entrenamiento

## Problema

Cuando se entrena con **muchas imágenes** (ej. 190+ imágenes .tif grandes), la RAM se agota porque:

- ❌ Se cargaban **todas las imágenes en memoria** de una vez
- ❌ Se convertían a tensores PyTorch antes de entrenar
- ❌ No se liberaba memoria entre batches

## Solución Implementada

### 1. **Lazy Loading Dataset** 🔄

**Archivo**: `backend/models/procesamiento.py`

Se creó `LazyImageDataset` que implementa carga perezosa:

```python
class LazyImageDataset:
    """
    Dataset que carga imágenes SOLO cuando se necesitan.
    No carga todo en memoria.
    """
    def __getitem__(self, idx):
        # Carga la imagen solo cuando se solicita
        img, _ = load_image(self.image_paths[idx])

        # Redimensionar al patch_size
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1)
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0),
            size=(self.patch_size, self.patch_size),
            mode='bilinear'
        ).squeeze(0)

        # Normalizar a [0, 1]
        img_min = img_tensor.min()
        img_max = img_tensor.max()
        img_tensor = (img_tensor - img_min) / (img_max - img_min + 1e-8)

        return img_tensor
```

**Beneficios**:

- ✅ Solo carga las imágenes del batch actual
- ✅ Libera memoria automáticamente después de procesar cada batch
- ✅ Soporta datasets ilimitados

### 2. **Liberación Explícita de Memoria** 🧹

**Archivo**: `backend/controllers/unsupervised_controller.py`

Se agregó limpieza de memoria después de cada batch:

```python
# Después de cada batch de entrenamiento
del xb, out, loss
if device.type == 'cuda':
    torch.cuda.empty_cache()
```

**Beneficios**:

- ✅ Libera VRAM de GPU inmediatamente
- ✅ Libera RAM del sistema
- ✅ Evita acumulación de tensores no usados

### 3. **DataLoader Optimizado** ⚙️

Se configuró DataLoader con parámetros óptimos para memoria:

```python
train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,  # Procesar pocas imágenes a la vez
    shuffle=True,
    num_workers=0,  # Sin multiprocessing (evita duplicar memoria)
    pin_memory=True  # Acelera transferencias a GPU
)
```

**Beneficios**:

- ✅ `num_workers=0`: Evita crear copias de datos en memoria
- ✅ `pin_memory=True`: Acelera CPU→GPU sin usar más RAM

### 4. **Train/Val Split sin Cargar Datos** 📊

**Antes**:

```python
# ❌ Cargaba todo en memoria primero
x_data = load_images_only(images_folder)  # 190 imágenes × 50MB = 9.5GB RAM!
x_train, x_val = train_test_split(x_data)
```

**Ahora**:

```python
# ✅ Solo obtiene las rutas, no carga imágenes
image_paths = get_image_paths(images_folder)  # Solo strings!
full_dataset = LazyImageDataset(image_paths)
train_ds, val_ds = random_split(full_dataset)  # Split sin cargar
```

### 5. **Cálculo de Threshold Optimizado** 🎯

**Antes**:

```python
# ❌ Convertía todo el dataset de validación a NumPy
recon_arr = np.concatenate(reconstructions)  # Gran array en RAM
```

**Ahora**:

```python
# ✅ Calcula error por batch y libera memoria
for batch in val_loader:
    error = torch.mean((xb - out) ** 2, dim=(1,2,3))
    reconstruction_errors.extend(error.cpu().numpy())
    del xb, out, error  # Liberar inmediatamente
```

---

## Comparación de Uso de Memoria

### Método Antiguo (Load All)

```
Ejemplo: 190 imágenes de 512×512×3, 8-bit

Memoria en load_images_only():
- NumPy array: 190 × 512 × 512 × 3 × 4 bytes = ~450 MB

Memoria después de train_test_split():
- x_train (80%): ~360 MB
- x_val (20%): ~90 MB

Memoria después de convertir a tensores:
- x_train_t: ~360 MB
- x_val_t: ~90 MB
- Arrays originales: ~450 MB (si no se liberan)

TOTAL: ~900 MB - 1.2 GB solo para datos
```

### Método Nuevo (Lazy Loading)

```
Memoria inicial:
- image_paths (lista de strings): ~10 KB
- LazyImageDataset (metadata): ~1 KB

Memoria durante entrenamiento (batch_size=8):
- batch actual en RAM: 8 × 512 × 512 × 3 × 4 = ~12 MB
- batch en GPU: ~12 MB
- Después de liberar: ~0 MB

TOTAL: ~12-24 MB durante entrenamiento
```

**🎯 Reducción de memoria: ~98%**

---

## Configuración Recomendada

### Para GPUs con poca VRAM (< 8GB):

```python
batch_size = 4  # Menos imágenes por batch
```

### Para GPUs con VRAM moderada (8-12GB):

```python
batch_size = 8  # Balance entre velocidad y memoria
```

### Para GPUs con mucha VRAM (> 12GB):

```python
batch_size = 16  # Máxima velocidad
```

---

## Funciones Añadidas

### `get_image_paths(images_folder) → List[str]`

Obtiene rutas de todas las imágenes .tif sin cargarlas.

**Uso**:

```python
paths = procesamiento.get_image_paths("C:/imagenes/")
print(f"Encontradas {len(paths)} imágenes")
```

### `LazyImageDataset`

Dataset de PyTorch con carga diferida.

**Uso**:

```python
dataset = procesamiento.LazyImageDataset(
    image_paths=paths,
    patch_size=128,
    cancel_check_fn=check_cancelled
)

loader = DataLoader(dataset, batch_size=8, shuffle=True)
for batch in loader:
    # Batch se carga aquí, no antes
    train(batch)
```

---

## Verificación

Para verificar que funciona correctamente:

1. **Antes de entrenar**:

   ```python
   import psutil
   print(f"RAM antes: {psutil.virtual_memory().percent}%")
   ```

2. **Durante entrenamiento**: Monitorear con Task Manager (Windows) o `nvidia-smi` (GPU)

3. **Después de liberar**:
   ```python
   torch.cuda.empty_cache()  # Liberar GPU
   print(f"RAM después: {psutil.virtual_memory().percent}%")
   ```

---

## Changelog

### 2025-11-10

- ✅ Agregado resize automático a `patch_size` en `LazyImageDataset`
- ✅ Implementada normalización min-max a [0,1] en `__getitem__`
- ✅ Fixed: Conversión de rutas Windows a WSL en `load_image()` y `load_mask()`
- ✅ Fixed: Modelo dinámico según patch_size en `ConvAutoencoder`
- ✅ Probado: Entrenamiento exitoso con 190 imágenes
- ✅ Probado: Inferencia exitosa en imagen 3000×4000×3

### 2025-11-09

- ✅ Implementado `LazyImageDataset` para lazy loading
- ✅ Agregado `get_image_paths()` para obtener rutas sin cargar
- ✅ Actualizado `execute_unsupervised_training()` para usar lazy loading
- ✅ Agregada liberación explícita de memoria después de cada batch
- ✅ Optimizado cálculo de threshold para no acumular tensores
- ✅ Configurado DataLoader con `num_workers=0` y `pin_memory=True`

---

## Notas Técnicas

- **PyTorch DataLoader**: Por defecto no libera memoria agresivamente. Es necesario usar `del` + `torch.cuda.empty_cache()`
- **num_workers=0**: En Windows, multiprocessing puede causar duplicación de memoria. Se desactiva para evitar esto
- **pin_memory**: Mejora velocidad de transferencia CPU→GPU sin incrementar uso de RAM significativamente
- **random_split**: Divide dataset sin duplicar datos, solo crea índices

---

## Próximas Mejoras Posibles

1. **Caché de imágenes frecuentes**: Guardar en RAM las imágenes más usadas
2. **Compresión on-the-fly**: Comprimir imágenes en RAM y descomprimir solo al usar
3. **Memory mapping**: Usar `np.memmap()` para archivos grandes
4. **Gradient accumulation**: Para batches más grandes sin más memoria
