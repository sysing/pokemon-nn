"""
Small CNN for Pokemon card type classification.
~600K parameters — fast training on M4 MPS.

Input:  [batch, 4, 224, 224]  (R, G, B, Luminance per pixel)
Output: [batch, 10]           (logits for 10 Pokemon types)

Includes spatial attention after conv layers to learn which card regions
(art area, text box, border) matter most for each type prediction.
"""

import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention: learns a [H,W] weight map over features."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        combined = torch.cat([avg, mx], dim=1)
        attn = self.conv(combined).sigmoid()
        return x * attn


class PokemonTypeCNN(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 4, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.spatial_attn = SpatialAttention()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.spatial_attn(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
