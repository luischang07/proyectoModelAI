"""
U-Net implementation in PyTorch for semantic segmentation
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Concatenate
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    https://arxiv.org/abs/1505.04597
    """
    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 1,
        features: list = [64, 128, 256, 512],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        
        # Input conv
        self.inc = DoubleConv(in_channels, features[0])
        
        # Encoder (downsampling)
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        
        # Bottleneck
        self.down4 = Down(features[3], features[3] * 2)
        
        # Decoder (upsampling)
        self.up1 = Up(features[3] * 2, features[3])
        self.up2 = Up(features[3], features[2])
        self.up3 = Up(features[2], features[1])
        self.up4 = Up(features[1], features[0])
        
        # Output conv
        self.outc = nn.Conv2d(features[0], n_classes, kernel_size=1)
        
        # Activation
        self.activation = nn.Sigmoid() if n_classes == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        x = self.outc(x)
        x = self.activation(x)
        return x


def build_unet(
    input_shape: Tuple[int, int, int],
    n_classes: int = 1,
    backbone: Optional[str] = None,
    encoder_weights: Optional[str] = None,
) -> UNet:
    """
    Crea un modelo U-Net en PyTorch
    
    Parámetros:
      - input_shape: (canales, alto, ancho) - formato PyTorch (C, H, W)
      - n_classes: número de clases de salida (1 para binaria con sigmoid)
      - backbone: ignorado (mantener compatibilidad con API anterior)
      - encoder_weights: ignorado (mantener compatibilidad con API anterior)
    
    Returns:
        UNet model
    """
    # input_shape puede venir como (C, H, W) formato PyTorch o (H, W, C) legacy
    # Detectar basándose en cual dimensión es más pequeña (canales suelen ser < 20)
    if len(input_shape) == 3:
        if input_shape[0] <= 16:  # Primer elemento es canales (formato PyTorch)
            in_channels = input_shape[0]
        elif input_shape[2] <= 16:  # Último elemento es canales (formato legacy)
            in_channels = input_shape[2]
        else:
            # Default a formato PyTorch si ambos son grandes
            in_channels = input_shape[0]
    else:
        in_channels = input_shape[0]
    
    model = UNet(in_channels=in_channels, n_classes=n_classes)
    return model


__all__ = ["build_unet", "UNet"]