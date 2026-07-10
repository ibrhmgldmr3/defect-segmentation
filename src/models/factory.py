"""segmentation-models-pytorch uzerinden U-Net / DeepLabV3+ model secici."""
import segmentation_models_pytorch as smp

_BUILDERS = {
    "unet": smp.Unet,
    "deeplabv3plus": smp.DeepLabV3Plus,
}


def build_model(name, encoder, encoder_weights="imagenet", in_channels=3, classes=1):
    try:
        builder = _BUILDERS[name]
    except KeyError:
        raise ValueError(f"Bilinmeyen model: {name}. Secenekler: {list(_BUILDERS)}")
    return builder(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
