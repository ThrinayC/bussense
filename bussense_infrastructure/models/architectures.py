"""
bussense_infrastructure/models/architectures.py
================================================
Lightweight, real PyTorch neural network architectures designed for
CPU-optimized edge execution in the BusSense pipeline.

Models:
1. LightweightRoadSegNet: Binary segmentation model for Waterlogging (FRED)
   and Zebra Crossing (CeyMo) detection.
2. TrafficSignNet: Detection & classification model for Traffic Signs (MTSVD/GTSDB).
3. SignConditionNet: Fine-grained physical condition classifier (NORMAL, DAMAGED, OCCLUDED).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class LightweightRoadSegNet(nn.Module):
    """
    Lightweight U-Net architecture optimized for CPU road perception.
    Input: [B, 3, H, W]
    Output: [B, 1, H, W] logits
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ConvBlock(32, 64)
        self.pool3 = nn.MaxPool2d(2)
        
        self.bottleneck = ConvBlock(64, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64, 32)
        
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(32, 16)
        
        self.out_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        b = self.bottleneck(self.pool3(e3))
        
        d3 = self.up3(b)
        if d3.shape != e3.shape:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        return self.out_conv(d1)


class TrafficSignNet(nn.Module):
    """
    Traffic sign classification and bounding box estimation network.
    Input: [B, 3, H, W]
    Outputs:
      - logits: [B, num_classes]
      - bbox: [B, 4] normalized [cx, cy, w, h]
    """
    CLASSES = ["SPEED_LIMIT", "STOP", "WARNING", "MANDATORY", "PROHIBITORY", "OTHER"]

    def __init__(self, in_channels: int = 3, num_classes: int = 6):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
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
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        self.bbox_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.features(x)
        cls_logits = self.classifier(feat)
        bbox = self.bbox_head(feat)
        return cls_logits, bbox


class SignConditionNet(nn.Module):
    """
    Physical condition classifier for traffic sign ROI crops.
    Input: [B, 3, 64, 64]
    Output: [B, 3] logits for [NORMAL, DAMAGED, OCCLUDED]
    """
    CONDITIONS = ["NORMAL", "DAMAGED", "OCCLUDED"]

    def __init__(self, in_channels: int = 3, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        return self.classifier(feat)
