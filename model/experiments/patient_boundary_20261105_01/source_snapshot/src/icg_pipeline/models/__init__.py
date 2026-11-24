"""U-Net model implementation for ICG segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    """Double convolution block with batch normalization and ReLU."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        mid_channels: Optional[int] = None,
        use_batchnorm: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        
        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=not use_batchnorm),
        ]
        
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(mid_channels))
        
        layers.append(nn.ReLU(inplace=True))
        
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        layers.extend([
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm),
        ])
        
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        layers.append(nn.ReLU(inplace=True))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """Downsampling block with MaxPool and DoubleConv."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        use_batchnorm: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_batchnorm=use_batchnorm, dropout=dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    """Upsampling block with ConvTranspose2d and DoubleConv."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
        bilinear: bool = False
    ):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, use_batchnorm, dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, use_batchnorm=use_batchnorm, dropout=dropout)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """U-Net architecture for ICG fluorescence segmentation.
    
    Args:
        in_channels: Number of input channels (default: 1 for grayscale ICG images)
        out_channels: Number of output channels (default: 2 for background/foreground)
        base_filters: Number of filters in the first layer (default: 64)
        depth: Depth of the U-Net (default: 4)
        use_batchnorm: Whether to use batch normalization (default: True)
        dropout: Dropout rate (default: 0.1)
        bilinear: Whether to use bilinear upsampling instead of transposed conv (default: False)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        base_filters: int = 64,
        depth: int = 4,
        use_batchnorm: bool = True,
        dropout: float = 0.1,
        bilinear: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        self.depth = depth
        self.bilinear = bilinear
        
        # Initial convolution
        self.inc = DoubleConv(in_channels, base_filters, use_batchnorm=use_batchnorm, dropout=dropout)
        
        # Encoder (Contracting Path)
        self.downs = nn.ModuleList()
        filters = base_filters
        for i in range(depth):
            self.downs.append(Down(filters, filters * 2, use_batchnorm=use_batchnorm, dropout=dropout))
            filters *= 2
        
        # Decoder (Expanding Path)
        self.ups = nn.ModuleList()
        for i in range(depth):
            self.ups.append(Up(filters, filters // 2, use_batchnorm=use_batchnorm, dropout=dropout, bilinear=bilinear))
            filters //= 2
        
        # Final output projection
        self.outc = OutConv(filters, out_channels)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"U-Net initialized: in_channels={in_channels}, out_channels={out_channels}, "
                   f"base_filters={base_filters}, depth={depth}, params={self.count_parameters():,}")
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder & Save Skip Connections
        skip_connections = []
        
        x = self.inc(x)
        skip_connections.append(x)
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
        
        # Decoder & Concatenate Skips
        skip_connections = skip_connections[:-1]  # Remove the bottleneck from skips
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        
        for i, up in enumerate(self.ups):
            x = up(x, skip_connections[i])
        
        logits = self.outc(x)
        return logits
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size(self) -> float:
        """Get model size in MB."""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / 1024 / 1024


class UNetWithAttention(UNet):
    """U-Net with Attention Gates for better feature selection."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add attention gates
        self.attention_gates = nn.ModuleList()
        filters = self.base_filters * (2 ** self.depth)
        for i in range(self.depth):
            self.attention_gates.append(AttentionGate(filters // 2, filters // 2, filters // 4))
            filters //= 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skip_connections = []
        x = self.inc(x)
        skip_connections.append(x)
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
        
        # Decoder with attention
        skip_connections = skip_connections[:-1][::-1]
        
        for i, up in enumerate(self.ups):
            # Apply attention gate
            skip = skip_connections[i]
            gating_signal = x
            skip = self.attention_gates[i](skip, gating_signal)
            x = up(x, skip)
        
        logits = self.outc(x)
        return logits


class AttentionGate(nn.Module):
    """Attention Gate for U-Net skip connections."""
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


def create_model(config: dict, load_pretrained: bool = True) -> nn.Module:
    """Factory function to create U-Net model from config."""
    model_config = config['model']
    name = model_config.get('name', 'unet')
    if name == 'unet_resnet18':
        from .resnet_unet import ResNetUNet
        return ResNetUNet(
            in_channels=model_config.get('in_channels', 3),
            out_channels=model_config.get('out_channels', 2),
            encoder_weights=model_config.get('encoder_weights', 'IMAGENET1K_V1') if load_pretrained else None,
            freeze_encoder_batchnorm=model_config.get('freeze_encoder_batchnorm', True),
            dropout=model_config.get('dropout', .1))
    if name != 'unet':
        raise ValueError(f'Unknown model: {name}')
    
    if model_config.get('use_attention', False):
        model = UNetWithAttention(
            in_channels=model_config['in_channels'],
            out_channels=model_config['out_channels'],
            base_filters=model_config['base_filters'],
            depth=model_config['depth'],
            use_batchnorm=model_config.get('use_batchnorm', True),
            dropout=model_config.get('dropout', 0.1),
            bilinear=model_config.get('bilinear', False)
        )
    else:
        model = UNet(
            in_channels=model_config['in_channels'],
            out_channels=model_config['out_channels'],
            base_filters=model_config['base_filters'],
            depth=model_config['depth'],
            use_batchnorm=model_config.get('use_batchnorm', True),
            dropout=model_config.get('dropout', 0.1),
            bilinear=model_config.get('bilinear', False)
        )
    
    return model
