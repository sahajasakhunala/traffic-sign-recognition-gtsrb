import torch.nn as nn
import torchvision.models as tv_models

def get_resnet50(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Returns a ResNet-50 model configured for the specified number of classes."""
    weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
    model = tv_models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
