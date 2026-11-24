"""Custom segmentation networks retained from the original other_models.py."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# --- 辅助模块 (所有模型共用) ---

class ConvBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # 调整卷积输入通道: 上采样通道 + skip连接通道
        self.conv_bn_relu1 = ConvBatchNormRelu(in_channels // 2 + skip_channels, out_channels)
        self.conv_bn_relu2 = ConvBatchNormRelu(out_channels, out_channels)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            # 确保空间维度匹配
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
            x = torch.cat([x, skip], dim=1)
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        return x

# --- 模型定义 ---

# 1. ResNet-UNet
# 1. ResNet-UNet (修正后)
class ResNetUNet(nn.Module):
    def __init__(self, in_channels=12, num_classes=2, pretrained_backbone=False): # <--- 修改参数名
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained_backbone else None) # <--- 相应修改
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder1 = nn.Sequential(self.first_conv, resnet.bn1, resnet.relu)
        self.encoder2 = resnet.layer1 # 64
        self.encoder3 = resnet.layer2 # 128
        self.encoder4 = resnet.layer3 # 256
        self.encoder5 = resnet.layer4 # 512
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 64)
        self.final_upsample = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            ConvBatchNormRelu(32, 32),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc1_pooled = self.pool(enc1)
        enc2 = self.encoder2(enc1_pooled)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        dec4 = self.decoder4(enc5, enc4)
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)
        out = self.final_upsample(dec1)
        out = self.final_conv(out)
        return out

# 2. SegNet
# 2. SegNet (MPS Compatible Version)
class SegNet(nn.Module):
    def __init__(self, in_channels=12, num_classes=2):
        super().__init__()
        # 编码器部分保持不变，但不再需要 return_indices=True
        self.enc_conv1 = nn.Sequential(ConvBatchNormRelu(in_channels, 64), ConvBatchNormRelu(64, 64))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 移除 return_indices
        self.enc_conv2 = nn.Sequential(ConvBatchNormRelu(64, 128), ConvBatchNormRelu(128, 128))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_conv3 = nn.Sequential(ConvBatchNormRelu(128, 256), ConvBatchNormRelu(256, 256), ConvBatchNormRelu(256, 256))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_conv4 = nn.Sequential(ConvBatchNormRelu(256, 512), ConvBatchNormRelu(512, 512), ConvBatchNormRelu(512, 512))
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_conv5 = nn.Sequential(ConvBatchNormRelu(512, 512), ConvBatchNormRelu(512, 512), ConvBatchNormRelu(512, 512))
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 解码器部分：用 Upsample 替换 Unpool
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv5 = nn.Sequential(ConvBatchNormRelu(512, 512), ConvBatchNormRelu(512, 512), ConvBatchNormRelu(512, 512))
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv4 = nn.Sequential(ConvBatchNormRelu(512, 512), ConvBatchNormRelu(512, 512), ConvBatchNormRelu(512, 256))
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv3 = nn.Sequential(ConvBatchNormRelu(256, 256), ConvBatchNormRelu(256, 256), ConvBatchNormRelu(256, 128))
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv2 = nn.Sequential(ConvBatchNormRelu(128, 128), ConvBatchNormRelu(128, 64))
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv1 = nn.Sequential(ConvBatchNormRelu(64, 64), ConvBatchNormRelu(64, num_classes))

    def forward(self, x):
        # 编码过程
        x1 = self.enc_conv1(x); p1 = self.pool1(x1)
        x2 = self.enc_conv2(p1); p2 = self.pool2(x2)
        x3 = self.enc_conv3(p2); p3 = self.pool3(x3)
        x4 = self.enc_conv4(p3); p4 = self.pool4(x4)
        x5 = self.enc_conv5(p4); p5 = self.pool5(x5)

        # 解码过程 (不再需要 indices 和 output_size)
        up5 = self.up5(p5); dec5 = self.dec_conv5(up5)
        up4 = self.up4(dec5); dec4 = self.dec_conv4(up4)
        up3 = self.up3(dec4); dec3 = self.dec_conv3(up3)
        up2 = self.up2(dec3); dec2 = self.dec_conv2(up2)
        up1 = self.up1(dec2); dec1 = self.dec_conv1(up1)

        return dec1
# 3. DeepLabv3+
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = [nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))]
        rates = tuple(atrous_rates)
        for rate in rates: modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout(0.5))
    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(nn.Conv2d(low_level_channels, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.aspp = ASPP(in_channels, aspp_dilate, out_channels=256)
        self.classifier = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, num_classes, 1))
    def forward(self, features):
        high_level_features, low_level_features = features['out'], features['low_level']
        low_level_features = self.project(low_level_features)
        aspp_output = self.aspp(high_level_features)
        high_features = F.interpolate(aspp_output, size=low_level_features.shape[2:], mode='bilinear', align_corners=False)
        concat_features = torch.cat([high_features, low_level_features], dim=1)
        return self.classifier(concat_features)

class DeepLabv3Plus(nn.Module):
    def __init__(self, in_channels=12, num_classes=2, pretrained_backbone=False):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained_backbone else None, replace_stride_with_dilation=[False, False, True])
        self.backbone_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone_layer1_base = nn.Sequential(resnet.bn1, resnet.relu, resnet.maxpool)
        self.backbone_layer1_blocks = resnet.layer1 # 256
        self.backbone_layer2 = resnet.layer2 # 512
        self.backbone_layer3 = resnet.layer3 # 1024
        self.backbone_layer4 = resnet.layer4 # 2048
        self.head = DeepLabHeadV3Plus(2048, 256, num_classes)

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = {}
        x = self.backbone_conv1(x)
        x = self.backbone_layer1_base(x)
        x = self.backbone_layer1_blocks(x)
        features['low_level'] = x
        x = self.backbone_layer2(x)
        x = self.backbone_layer3(x)
        x = self.backbone_layer4(x)
        features['out'] = x
        output = self.head(features)
        output = F.interpolate(output, size=input_shape, mode='bilinear', align_corners=False)
        return output

# 4. RDU-UNet
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBatchNormRelu(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x; out = self.conv1(x); out = self.bn2(self.conv2(out)); out += residual; return self.relu(out)

class DenseLayer(nn.Module):
     def __init__(self, in_channels, growth_rate):
          super().__init__(); self.H_l = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True), nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False))
     def forward(self, x): return torch.cat([x, self.H_l(x)], 1)

class DenseBlock(nn.Module):
     def __init__(self, in_channels, num_layers, growth_rate):
        super().__init__(); layers = []; current_channels = in_channels
        for i in range(num_layers): layers.append(DenseLayer(current_channels, growth_rate)); current_channels += growth_rate
        self.block = nn.Sequential(*layers); self.output_channels = current_channels
     def forward(self, x): return self.block(x)

class RDU_UNet(nn.Module):
    def __init__(self, in_channels=12, num_classes=2, growth_rate=32, num_layers_dense=4):
        super().__init__()
        self.init_conv = ConvBatchNormRelu(in_channels, 64)
        self.enc_res1 = ResidualBlock(64); self.pool1 = nn.MaxPool2d(2)
        self.enc_res2 = ResidualBlock(64); self.pool2 = nn.MaxPool2d(2)
        self.enc_res3 = ResidualBlock(64); self.pool3 = nn.MaxPool2d(2)
        self.dense_bottleneck = DenseBlock(64, num_layers_dense, growth_rate)
        bottleneck_out_channels = self.dense_bottleneck.output_channels
        self.up3 = nn.ConvTranspose2d(bottleneck_out_channels, 128, kernel_size=2, stride=2)
        self.dec_conv3 = nn.Sequential(ConvBatchNormRelu(128 + 64, 128), ConvBatchNormRelu(128, 64))
        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Sequential(ConvBatchNormRelu(64 + 64, 64), ConvBatchNormRelu(64, 64))
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Sequential(ConvBatchNormRelu(64 + 64, 64), ConvBatchNormRelu(64, 32))
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        skip_init = self.init_conv(x)
        skip1 = self.enc_res1(skip_init); p1 = self.pool1(skip1)
        skip2 = self.enc_res2(p1); p2 = self.pool2(skip2)
        skip3 = self.enc_res3(p2); p3 = self.pool3(skip3)
        bottleneck = self.dense_bottleneck(p3)
        d3 = self.up3(bottleneck); d3 = torch.cat([d3, skip3], dim=1); d3 = self.dec_conv3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, skip2], dim=1); d2 = self.dec_conv2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, skip_init], dim=1); d1 = self.dec_conv1(d1)
        output = self.final_conv(d1)
        return output

# 5. SimpleUnet
class SimpleUnet(nn.Module):
    def __init__(self, in_channels=12, num_classes=2):
        super(SimpleUnet, self).__init__()
        self.enc_conv1 = nn.Sequential(ConvBatchNormRelu(in_channels, 64), ConvBatchNormRelu(64, 64)); self.pool1 = nn.MaxPool2d(2, 2)
        self.enc_conv2 = nn.Sequential(ConvBatchNormRelu(64, 128), ConvBatchNormRelu(128, 128)); self.pool2 = nn.MaxPool2d(2, 2)
        self.enc_conv3 = nn.Sequential(ConvBatchNormRelu(128, 256), ConvBatchNormRelu(256, 256)) # Bottleneck
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Sequential(ConvBatchNormRelu(256, 128), ConvBatchNormRelu(128, 128))
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Sequential(ConvBatchNormRelu(128, 64), ConvBatchNormRelu(64, 64))
        self.output_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc_conv1(x); p1 = self.pool1(enc1)
        enc2 = self.enc_conv2(p1); p2 = self.pool2(enc2)
        bottleneck = self.enc_conv3(p2)
        dec2 = self.up2(bottleneck)
        diffY = enc2.size()[2] - dec2.size()[2]; diffX = enc2.size()[3] - dec2.size()[3]
        dec2 = F.pad(dec2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        cat2 = torch.cat([dec2, enc2], dim=1); dec2_out = self.dec_conv2(cat2)
        dec1 = self.up1(dec2_out)
        diffY = enc1.size()[2] - dec1.size()[2]; diffX = enc1.size()[3] - dec1.size()[3]
        dec1 = F.pad(dec1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        cat1 = torch.cat([dec1, enc1], dim=1); dec1_out = self.dec_conv1(cat1)
        output = self.output_conv(dec1_out)
        return output

# 6. DeepLabv3 (Simplified)
class DeepLabv3(nn.Module):
    def __init__(self, in_channels=12, num_classes=2):
        super(DeepLabv3, self).__init__()
        # Simplified backbone
        self.backbone = nn.Sequential(
            ConvBatchNormRelu(in_channels, 64, stride=2),
            ConvBatchNormRelu(64, 128, stride=2)
        )
        # Simplified ASPP
        self.aspp = ASPP(128, atrous_rates=[6, 12, 18], out_channels=256)
        self.final_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.final_conv(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

# 7. SimpleCNN
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=12, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = ConvBatchNormRelu(in_channels, 32); self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = ConvBatchNormRelu(32, 64); self.pool2 = nn.MaxPool2d(2, 2)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2); self.conv_up1 = ConvBatchNormRelu(32, 32)
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2); self.conv_up2 = ConvBatchNormRelu(16, 16)
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.pool1(self.conv1(x)); x = self.pool2(self.conv2(x))
        x = self.conv_up1(self.up1(x)); x = self.conv_up2(self.up2(x))
        x = self.final_conv(x)
        return x