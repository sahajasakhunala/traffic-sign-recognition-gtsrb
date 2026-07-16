import torch.nn as nn
import torchvision.models as tv_models

def get_efficientnet_b0(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Returns an EfficientNet-B0 model configured for the specified number of classes."""
    weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = tv_models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def get_efficientnet_v2_s(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Returns an EfficientNet-V2-S model configured for the specified number of classes."""
    weights = tv_models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    model = tv_models.efficientnet_v2_s(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
