"""ResNet-18 encoder and a small U-Net decoder; torchvision is imported lazily."""
import torch
from torch import nn
from torch.nn import functional as F

from . import DoubleConv


class ResNetUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, encoder_weights='IMAGENET1K_V1',
                 freeze_encoder_batchnorm=True, dropout=0.1):
        super().__init__()
        if in_channels != 3 or out_channels != 2:
            raise ValueError('unet_resnet18 requires RGB input and two-class output')
        try:
            from torchvision.models import resnet18, ResNet18_Weights
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError('Pipeline B requires torchvision matching your torch installation; '
                               'install model/requirements-resnet.txt.') from exc
        if encoder_weights not in (None, 'IMAGENET1K_V1'):
            raise ValueError('encoder_weights must be IMAGENET1K_V1 or null (explicit scratch experiment)')
        try:
            self.encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if encoder_weights else None)
        except Exception as exc:
            raise RuntimeError('Could not load ResNet18 IMAGENET1K_V1 weights. Download/cache the '
                               'Torchvision weights on a connected machine; random fallback is disabled.') from exc
        self.encoder.fc = nn.Identity()
        self.freeze_encoder_batchnorm = freeze_encoder_batchnorm
        # Encoder skips: layer3=256, layer2=128, layer1=64, stem=64.
        self.decoder = nn.ModuleList([
            DoubleConv(512+256, 256, dropout=dropout),
            DoubleConv(256+128, 128, dropout=dropout),
            DoubleConv(128+64, 64, dropout=dropout),
            DoubleConv(64+64, 32, dropout=dropout),
            DoubleConv(32, 32, dropout=dropout),
        ])
        self.output = nn.Conv2d(32, out_channels, 1)
        for module in list(self.decoder.modules()) + [self.output]:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder_batchnorm:
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, x):
        size = x.shape[-2:]
        e = self.encoder
        stem = e.relu(e.bn1(e.conv1(x)))
        l1 = e.layer1(e.maxpool(stem))
        l2 = e.layer2(l1)
        l3 = e.layer3(l2)
        z = e.layer4(l3)
        for block, skip in zip(self.decoder[:4], (l3, l2, l1, stem)):
            z = F.interpolate(z, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            z = block(torch.cat((z, skip), dim=1))
        z = self.decoder[4](F.interpolate(z, size=size, mode='bilinear', align_corners=False))
        return self.output(z)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size(self):
        return sum(p.numel()*p.element_size() for p in list(self.parameters()) + list(self.buffers())) / 1024**2
