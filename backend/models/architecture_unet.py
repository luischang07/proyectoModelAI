import os

# Configurar ANTES de cualquier import de TensorFlow/Keras
os.environ.setdefault("SM_FRAMEWORK", "tf.keras")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TensorFlow logging

from typing import Optional, Tuple
import segmentation_models as sm

# Asegurar que segmentation-models use tf.keras
sm.set_framework("tf.keras")


def build_unet(
    input_shape: Tuple[int, int, int],
    n_classes: int = 1,
    backbone: Optional[str] = None,
    encoder_weights: Optional[str] = None,
):
    """Crea y compila un modelo U-Net con segmentation-models.

    Parámetros:
      - input_shape: (patch, patch, canales)
      - n_classes: número de clases de salida (1 para binaria con sigmoid)
      - backbone: nombre del encoder, p.ej. 'resnet34' o None para un U-Net básico
      - encoder_weights: 'imagenet' o None; usar None si canales != 3
    """
    # Por compatibilidad, si no se especifica backbone, usar 'resnet34'
    backbone = backbone or "resnet34"

    model = sm.Unet(
        backbone,
        encoder_weights=encoder_weights,
        input_shape=input_shape,
        classes=n_classes,
        activation="sigmoid" if n_classes == 1 else "softmax",
    )

    loss_function = sm.losses.DiceLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    metric = sm.metrics.IOUScore(threshold=0.5)

    # Usar identificador de optimizador para evitar dependencia directa en TF en análisis
    model.compile(optimizer="adam", loss=loss_function, metrics=[metric])
    return model


__all__ = ["build_unet"]