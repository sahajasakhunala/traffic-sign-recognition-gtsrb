import torch.nn as nn
import torchvision.models as tv_models

def get_mobilenet_v3_large(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Returns a MobileNet-V3-Large model configured for the specified number of classes."""
    weights = tv_models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = tv_models.mobilenet_v3_large(weights=weights)
    # MobileNetV3 Large classifier head consists of a Dropout, Linear, Hardswish, and Linear layer.
    # The last Linear layer is at index 3.
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model
