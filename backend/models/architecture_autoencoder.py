"""
Autoencoder para detección de anomalías NO SUPERVISADO

Este modelo aprende a reconstruir imágenes normales (sanas).
Las anomalías se detectan por alto error de reconstrucción.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from keras import layers


def build_autoencoder(
    input_shape: Tuple[int, int, int],
    latent_dim: int = 128,
    encoder_filters: list = None,
):
    """
    Crea un Autoencoder convolucional para detección de anomalías.
    
    Args:
        input_shape: (patch_size, patch_size, channels)
        latent_dim: Dimensión del espacio latente (bottleneck)
        encoder_filters: Lista de filtros por capa [64, 128, 256]
    
    Returns:
        tuple: (autoencoder_model, encoder_model, decoder_model)
    """
    
    if encoder_filters is None:
        encoder_filters = [64, 128, 256]
    
    # ========== ENCODER ==========
    encoder_input = layers.Input(shape=input_shape, name="encoder_input")
    x = encoder_input
    
    # Capas convolucionales del encoder
    for i, filters in enumerate(encoder_filters):
        x = layers.Conv2D(
            filters, 
            kernel_size=3, 
            strides=2, 
            padding="same",
            name=f"encoder_conv_{i+1}"
        )(x)
        x = layers.BatchNormalization(name=f"encoder_bn_{i+1}")(x)
        x = layers.LeakyReLU(alpha=0.2, name=f"encoder_relu_{i+1}")(x)
    
    # Bottleneck (espacio latente)
    shape_before_flatten = x.shape[1:]  # Guardar para el decoder
    x = layers.Flatten(name="encoder_flatten")(x)
    latent = layers.Dense(latent_dim, name="latent_space")(x)
    
    encoder = keras.Model(encoder_input, latent, name="encoder")
    
    # ========== DECODER ==========
    decoder_input = layers.Input(shape=(latent_dim,), name="decoder_input")
    
    # Reconstruir forma antes del flatten
    flatten_size = shape_before_flatten[0] * shape_before_flatten[1] * shape_before_flatten[2]
    x = layers.Dense(flatten_size, name="decoder_dense")(decoder_input)
    x = layers.Reshape(shape_before_flatten, name="decoder_reshape")(x)
    
    # Capas convolucionales transpuestas del decoder
    for i, filters in enumerate(reversed(encoder_filters[:-1])):
        x = layers.Conv2DTranspose(
            filters,
            kernel_size=3,
            strides=2,
            padding="same",
            name=f"decoder_conv_{i+1}"
        )(x)
        x = layers.BatchNormalization(name=f"decoder_bn_{i+1}")(x)
        x = layers.LeakyReLU(alpha=0.2, name=f"decoder_relu_{i+1}")(x)
    
    # Última capa para reconstruir la imagen original
    x = layers.Conv2DTranspose(
        input_shape[2],  # Mismo número de canales que la entrada
        kernel_size=3,
        strides=2,
        padding="same",
        activation="sigmoid",
        name="decoder_output"
    )(x)
    
    decoder = keras.Model(decoder_input, x, name="decoder")
    
    # ========== AUTOENCODER COMPLETO ==========
    autoencoder_output = decoder(encoder(encoder_input))
    autoencoder = keras.Model(encoder_input, autoencoder_output, name="autoencoder")
    
    return autoencoder, encoder, decoder


def compile_autoencoder(autoencoder: keras.Model, learning_rate: float = 0.001):
    """
    Compila el autoencoder con función de pérdida apropiada.
    
    Args:
        autoencoder: Modelo autoencoder
        learning_rate: Tasa de aprendizaje
    """
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",  # Mean Squared Error para reconstrucción
        metrics=["mae"]  # Mean Absolute Error como métrica adicional
    )
    return autoencoder


def detect_anomalies(
    autoencoder: keras.Model,
    images: tf.Tensor,
    threshold: float = None,
    percentile: float = 95
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Detecta anomalías usando error de reconstrucción.
    
    Args:
        autoencoder: Modelo entrenado
        images: Imágenes a evaluar (N, H, W, C)
        threshold: Umbral de error (si None, se calcula con percentile)
        percentile: Percentil para calcular umbral automático
    
    Returns:
        tuple: (errores_reconstruccion, máscaras_anomalías)
    """
    # Reconstruir imágenes
    reconstructed = autoencoder.predict(images, verbose=0)
    
    # Calcular error de reconstrucción por píxel
    reconstruction_errors = tf.reduce_mean(
        tf.square(images - reconstructed), 
        axis=-1  # Promedio sobre canales
    )
    
    # Calcular umbral si no se proporciona
    if threshold is None:
        threshold = tf.experimental.numpy.percentile(
            reconstruction_errors, 
            percentile
        )
    
    # Máscara binaria: 1 = anomalía, 0 = normal
    anomaly_masks = tf.cast(reconstruction_errors > threshold, tf.float32)
    
    return reconstruction_errors, anomaly_masks


__all__ = [
    "build_autoencoder",
    "compile_autoencoder", 
    "detect_anomalies"
]
