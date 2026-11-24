"""Custom timm U-Net and fixed-size input adapter, with original parameter names."""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.relu1(self.bn1(self.conv1(x)))
        return self.relu2(self.bn2(self.conv2(x)))


class TimmUnet(nn.Module):
    def __init__(self, encoder_name: str, in_channels: int = 3,
                 num_classes: int = 1, pretrained: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = timm.create_model(encoder_name, pretrained=pretrained,
                                         features_only=True, in_chans=in_channels)
        encoder_channels = self.encoder.feature_info.channels()
        reversed_channels = list(reversed(encoder_channels))
        self.bottleneck = nn.Sequential(
            nn.Conv2d(reversed_channels[0], reversed_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(reversed_channels[0]),
            nn.ReLU(inplace=True),
        )
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(reversed_channels) - 1):
            self.decoder_blocks.append(DecoderBlock(
                reversed_channels[i], reversed_channels[i + 1], reversed_channels[i + 1]))
        self.segmentation_head = nn.Conv2d(encoder_channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        expected_channels = self.encoder.feature_info.channels()
        if features[0].shape[-1] == expected_channels[0]:
            features = [feature.permute(0, 3, 1, 2) for feature in features]
        reversed_features = list(reversed(features))
        decoded = self.bottleneck(reversed_features[0])
        for i, block in enumerate(self.decoder_blocks):
            decoded = block(decoded, reversed_features[i + 1])
        logits = self.segmentation_head(decoded)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=True)
        return logits


class ModelInputSizeAdapter(nn.Module):
    def __init__(self, model_creator_fn, expected_input_size):
        super().__init__()
        self.model = model_creator_fn()
        self.expected_input_size = expected_input_size

    def forward(self, x):
        _, _, height, width = x.shape
        if height != self.expected_input_size or width != self.expected_input_size:
            resized = F.interpolate(x, size=(self.expected_input_size, self.expected_input_size),
                                    mode="bilinear", align_corners=False)
            output = self.model(resized)
            return F.interpolate(output, size=(height, width), mode="bilinear", align_corners=False)
        return self.model(x)
