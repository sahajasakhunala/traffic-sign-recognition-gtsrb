import torch
import torch.nn as nn
from typing import Optional

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention.
    
    Recalibrates channel-wise feature responses by explicitly modeling
    interdependencies between channels.
    """
    
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid_channels = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.squeeze(x).view(b, c)
        w = self.excite(w).view(b, c, 1, 1)
        return x * w


class ResidualBlock(nn.Module):
    """Residual convolution block with optional Squeeze-and-Excitation.
    
    Consists of two Conv2d->BN->SiLU operations with a shortcut connection.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_se: bool = True,
        se_reduction: int = 8
    ) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.SiLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU(inplace=True)
        
        self.se = SEBlock(out_channels, se_reduction) if use_se else nn.Identity()
        
        # Shortcut mapping if dimensions change
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        
        out += residual
        out = self.act2(out)
        return out


class CustomCNNV2(nn.Module):
    """Advanced Custom CNN (V2) for traffic sign classification.
    
    Features:
    - BatchNorm and SiLU activations for stable training.
    - Residual connections to prevent gradient degradation.
    - Squeeze-and-Excitation blocks for channel-wise attention.
    - Global Average Pooling (GAP) classifier head to minimize overfitting.
    """
    
    def __init__(self, num_classes: int = 43) -> None:
        super().__init__()
        
        # Stem block (conv downsample 64x64 -> 64x64)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Residual stages (downsampling by 2 at each stage)
        # 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.stage1 = ResidualBlock(32, 64, stride=2, use_se=True)
        self.stage2 = ResidualBlock(64, 128, stride=2, use_se=True)
        self.stage3 = ResidualBlock(128, 256, stride=2, use_se=True)
        self.stage4 = ResidualBlock(256, 512, stride=2, use_se=True)
        
        # Classifier head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights using Kaiming normal
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.pool(out)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    model = CustomCNNV2(num_classes=43)
    dummy = torch.randn(2, 3, 64, 64)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # Should be [2, 43]
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
