import torch.nn as nn
import torchvision.models as tv_models

def get_convnext_tiny(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Returns a ConvNeXt-Tiny model configured for the specified number of classes."""
    weights = tv_models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = tv_models.convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model
