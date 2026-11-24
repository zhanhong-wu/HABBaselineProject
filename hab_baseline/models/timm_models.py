"""Timm-based baseline builders."""

from .timm_unet import ModelInputSizeAdapter, TimmUnet

TIMM_MODEL_SPECS = {
    "Swin_Unet_tiny": ("swin_tiny_patch4_window7_224", 224),
    "Swin_Unet_small": ("swin_small_patch4_window7_224", 224),
    "ConvNeXt_Unet_tiny": ("convnext_tiny", None),
    "ConvNeXt_Unet_small": ("convnext_small", None),
    "MaxViT_Unet_tiny": ("maxvit_tiny_tf_224", 224),
}

TIMM_MODEL_NAMES = list(TIMM_MODEL_SPECS)


def build_timm_model(name, in_channels, num_classes):
    encoder, expected_size = TIMM_MODEL_SPECS[name]
    creator = lambda: TimmUnet(encoder_name=encoder, in_channels=in_channels,
                               num_classes=num_classes, pretrained=False)
    if expected_size is None:
        return creator()
    return ModelInputSizeAdapter(creator, expected_input_size=expected_size)
