"""Segmentation Models PyTorch builders used by the published baselines."""

import segmentation_models_pytorch as smp

from .timm_unet import ModelInputSizeAdapter

SMP_MODEL_SPECS = {
    "Unet_resnet34": (smp.Unet, "resnet34"),
    "UnetPlusPlus_resnet34": (smp.UnetPlusPlus, "resnet34"),
    "MAnet_resnet34": (smp.MAnet, "resnet34"),
    "Linknet_resnet34": (smp.Linknet, "resnet34"),
    "FPN_resnet34": (smp.FPN, "resnet34"),
    "PSPNet_resnet34": (smp.PSPNet, "resnet34"),
    "PAN_resnet34": (smp.PAN, "resnet34"),
    "DeepLabV3_resnet34": (smp.DeepLabV3, "resnet34"),
    "DeepLabV3Plus_resnet34": (smp.DeepLabV3Plus, "resnet34"),
    "UPerNet_resnet34": (smp.UPerNet, "resnet34"),
    "Segformer_mit_b0": (smp.Segformer, "mit_b0"),
}


def build_smp_model(name, in_channels, num_classes):
    if name == "DPT_vit_base":
        return ModelInputSizeAdapter(
            model_creator_fn=lambda: smp.DPT(
                encoder_name="tu-vit_base_patch16_224", encoder_weights=None,
                in_channels=in_channels, classes=num_classes),
            expected_input_size=224,
        )
    model_class, encoder = SMP_MODEL_SPECS[name]
    return model_class(encoder_name=encoder, encoder_weights=None,
                       in_channels=in_channels, classes=num_classes)


SMP_MODEL_NAMES = list(SMP_MODEL_SPECS) + ["DPT_vit_base"]
