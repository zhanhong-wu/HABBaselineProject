"""Unified model registry with legacy checkpoint-compatible names."""

from .custom import DeepLabv3, DeepLabv3Plus, RDU_UNet, ResNetUNet, SegNet, SimpleCNN, SimpleUnet
from .smp_models import SMP_MODEL_NAMES, build_smp_model
from .timm_models import TIMM_MODEL_NAMES, build_timm_model

BASELINE_MODEL_NAMES = [
    "Unet_resnet34",
    "UnetPlusPlus_resnet34",
    "MAnet_resnet34",
    "Linknet_resnet34",
    "FPN_resnet34",
    "PSPNet_resnet34",
    "PAN_resnet34",
    "DeepLabV3_resnet34",
    "DeepLabV3Plus_resnet34",
    "UPerNet_resnet34",
    "Segformer_mit_b0",
    "Swin_Unet_tiny",
    "ConvNeXt_Unet_tiny",
    "MaxViT_Unet_tiny",
    "DPT_vit_base",
]

CUSTOM_MODEL_BUILDERS = {
    "ResNetUNet_custom": ResNetUNet,
    "SegNet_custom": SegNet,
    "DeepLabV3Plus_custom": DeepLabv3Plus,
    "RDU_UNet_custom": RDU_UNet,
    "SimpleUnet_custom": SimpleUnet,
    "DeepLabV3_custom": DeepLabv3,
    "SimpleCNN_custom": SimpleCNN,
}


def available_models():
    return BASELINE_MODEL_NAMES + list(CUSTOM_MODEL_BUILDERS)


def build_model(name, in_channels=12, num_classes=2):
    if name in SMP_MODEL_NAMES:
        return build_smp_model(name, in_channels, num_classes)
    if name in TIMM_MODEL_NAMES:
        return build_timm_model(name, in_channels, num_classes)
    if name in CUSTOM_MODEL_BUILDERS:
        return CUSTOM_MODEL_BUILDERS[name](in_channels=in_channels, num_classes=num_classes)
    raise ValueError(f"Unknown model {name!r}. Available models: {', '.join(available_models())}")


def legacy_file_tag(model_name):
    custom_unet = {"Swin_Unet_tiny", "Swin_Unet_small", "ConvNeXt_Unet_tiny",
                   "ConvNeXt_Unet_small", "MaxViT_Unet_tiny", "DPT_vit_base"}
    return "custom_unet" if model_name in custom_unet else "from_scratch"
