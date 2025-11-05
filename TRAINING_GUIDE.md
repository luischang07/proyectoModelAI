# 🧠 Guía de Entrenamiento: Supervisado vs No Supervisado

## 📖 Resumen

Este proyecto ahora soporta **DOS tipos de entrenamiento**:

### 1️⃣ **Supervisado** (con máscaras etiquetadas)

- **Modelo**: U-Net con encoder preentrenado
- **Requiere**: Imágenes + Máscaras binarias etiquetadas
- **Uso**: Cuando tienes datos etiquetados (máscaras dibujadas)
- **Resultado**: Segmentación precisa de anomalías conocidas

### 2️⃣ **No Supervisado** (sin máscaras) ✨ **NUEVO**

- **Modelo**: Autoencoder convolucional
- **Requiere**: Solo imágenes (sin máscaras)
- **Uso**: Cuando NO tienes datos etiquetados
- **Resultado**: Detecta anomalías por error de reconstrucción

---

## 🎯 ¿Cuándo usar cada uno?

### Usa **SUPERVISADO** si:

- ✅ Tienes máscaras etiquetadas (anomalías marcadas manualmente)
- ✅ Sabes exactamente qué buscar
- ✅ Quieres alta precisión en segmentación
- ✅ Tienes tiempo para etiquetar datos

**Ejemplo**: Detectar áreas con plagas específicas que ya conoces

### Usa **NO SUPERVISADO** si:

- ✅ NO tienes máscaras etiquetadas
- ✅ No tienes tiempo/recursos para etiquetar
- ✅ Quieres detectar anomalías desconocidas
- ✅ Solo tienes imágenes "normales" disponibles

**Ejemplo**: Detectar cualquier anomalía nueva/desconocida en cultivos

---

## 🔬 Cómo funciona cada modelo

### Supervisado (U-Net)

```
Entrada: Imagen (RGB/multiespectral)
Salida: Máscara de segmentación (0=sano, 1=anomalía)

Entrenamiento:
  - Aprende de pares (imagen, máscara_etiquetada)
  - Minimiza diferencia entre predicción y máscara real
  - Requiere Ground Truth
```

### No Supervisado (Autoencoder)

```
Entrada: Imagen (RGB/multiespectral)
Salida: Imagen reconstruida + Error de reconstrucción

Entrenamiento:
  - Aprende a reconstruir imágenes NORMALES
  - Solo usa imágenes sin etiquetas
  - Comprime imagen → Reconstruye imagen

Inferencia:
  - Alto error de reconstrucción = ANOMALÍA
  - Bajo error de reconstrucción = NORMAL
```

**Arquitectura del Autoencoder:**

```
Encoder:          Latent Space:      Decoder:
[128x128x3]  →    [Compressed]   →   [128x128x3]
    ↓                                     ↑
  Conv2D                               Conv2DTranspose
    ↓                                     ↑
  Conv2D         [128 dims]           Conv2DTranspose
    ↓                                     ↑
  Conv2D                               Conv2DTranspose

Loss = MSE(Original, Reconstruida)
```

---

## 🚀 Uso en el Frontend

### Interfaz Actualizada

1. **Seleccionar Tipo de Entrenamiento**:

   - Dropdown con dos opciones
   - Se habilita/deshabilita campo de máscaras automáticamente

2. **Modo Supervisado**:

   - Campo "Imágenes": **Requerido** ✅
   - Campo "Máscaras": **Requerido** ✅
   - Parámetros: patch_size, stride, batch_size, epochs, backbone

3. **Modo No Supervisado**:
   - Campo "Imágenes": **Requerido** ✅
   - Campo "Máscaras": **Deshabilitado** ❌
   - Parámetros: batch_size, epochs, latent_dim, validation_split

---

## 📊 Comparación de Resultados

| Característica               | Supervisado               | No Supervisado         |
| ---------------------------- | ------------------------- | ---------------------- |
| **Precisión**                | ⭐⭐⭐⭐⭐ Alta           | ⭐⭐⭐ Media           |
| **Requiere etiquetas**       | ✅ Sí                     | ❌ No                  |
| **Tiempo de preparación**    | 🕐 Alto (etiquetar)       | ⚡ Rápido              |
| **Detecta anomalías nuevas** | ❌ Solo conocidas         | ✅ Sí                  |
| **Cantidad de datos**        | Media (100-1000 imágenes) | Alta (1000+ imágenes)  |
| **Interpretabilidad**        | ⭐⭐⭐⭐⭐ Muy clara      | ⭐⭐⭐ Necesita umbral |

---

## 🛠️ Ejemplo de Uso

### Supervisado (con máscaras)

```python
# En el frontend
1. Seleccionar "Supervisado (con máscaras)"
2. Carpeta imágenes: data/images/
3. Carpeta máscaras: data/masks/
4. Epochs: 25
5. Iniciar entrenamiento

# Resultado
✅ Modelo entrenado: models/unet_model.keras
✅ Métricas: IoU Score, Dice Loss
✅ Listo para segmentación precisa
```

### No Supervisado (sin máscaras)

```python
# En el frontend
1. Seleccionar "No Supervisado (sin máscaras - Autoencoder)"
2. Carpeta imágenes: data/images/  (solo imágenes normales)
3. Epochs: 50
4. Latent dim: 128
5. Iniciar entrenamiento

# Resultado
✅ Modelo entrenado: models/autoencoder_model.keras
✅ Umbral de anomalía: 0.0234 (calculado automáticamente)
✅ Listo para detección de anomalías
```

---

## 📁 Estructura de Datos

### Para Supervisado

```
data/
├── images/
│   ├── image_001.tif
│   ├── image_002.tif
│   └── image_003.tif
└── masks/
    ├── image_001.tif  (0=sano, 1=anomalía)
    ├── image_002.tif
    └── image_003.tif
```

### Para No Supervisado

```
data/
└── images/
    ├── normal_001.tif  (solo imágenes NORMALES)
    ├── normal_002.tif
    ├── normal_003.tif
    └── ...
```

⚠️ **Importante para No Supervisado**:

- Entrena **SOLO con imágenes NORMALES/SANAS**
- El modelo aprenderá qué es "normal"
- En inferencia, lo diferente a "normal" = anomalía

---

## 🎓 Referencias

### Supervisado (U-Net)

- Paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- Segmentation Models: https://github.com/qubvel/segmentation_models

### No Supervisado (Autoencoder)

- Paper: [Autoencoding beyond pixels using a learned similarity metric](https://arxiv.org/abs/1512.09300)
- Tutorial: [Anomaly Detection with Autoencoders](https://keras.io/examples/timeseries/timeseries_anomaly_detection/)

---

## 📝 Notas Técnicas

### Supervisado

```python
# Training
X: imágenes normalizadas (N, H, W, C)
y: máscaras binarias (N, H, W, 1)

# Loss
Loss = DiceLoss(y_true, y_pred)
Metric = IoUScore(y_true, y_pred)
```

### No Supervisado

```python
# Training
X: imágenes normalizadas (N, H, W, C)
y: mismas imágenes (N, H, W, C)  # Autoreconstrucción

# Loss
Loss = MSE(X, X_reconstructed)

# Inference
reconstruction_error = mean(square(X - X_reconstructed))
is_anomaly = reconstruction_error > threshold
```

---

## 🆘 FAQ

**P: ¿Puedo combinar ambos métodos?**
R: Sí! Puedes entrenar primero un Autoencoder (sin máscaras) para detectar candidatos, luego etiquetar solo esos candidatos y entrenar U-Net.

**P: ¿Cuál es más rápido de entrenar?**
R: El Autoencoder suele ser más rápido (sin preprocesar máscaras), pero requiere más épocas.

**P: ¿Cuál necesita más datos?**
R: No supervisado necesita MÁS datos (solo imágenes) porque no tiene señal de etiquetas.

**P: ¿Puedo usar el Autoencoder si tengo máscaras?**
R: Sí, pero es mejor usar el método Supervisado para aprovechar las etiquetas.

---

## 🎨 Próximas Mejoras

- [ ] Inferencia con Autoencoder (detectar anomalías en producción)
- [ ] Visualización de error de reconstrucción
- [ ] Ensemble: Autoencoder + U-Net
- [ ] Semi-supervisado (pocas máscaras + muchas imágenes)

---

**¡Ahora tienes flexibilidad total para entrenar con o sin máscaras!** 🎉
