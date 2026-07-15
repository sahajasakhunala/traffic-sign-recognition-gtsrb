import torch.nn as nn
from typing import Dict, Any

from models.cnn_v2 import CustomCNNV2
from models.efficientnet import get_efficientnet_b0
from models.resnet import get_resnet50
from models.convnext import get_convnext_tiny

def create_model(config: Dict[str, Any]) -> nn.Module:
    """Instantiates a model architecture based on the configuration dictionary.
    
    Supported architectures:
    - 'cnn_v2': Custom CNN with Residual blocks and Squeeze-and-Excitation.
    - 'efficientnet_b0': Pretrained/from-scratch EfficientNet-B0.
    - 'resnet50': Pretrained/from-scratch ResNet-50.
    - 'convnext_tiny': Pretrained/from-scratch ConvNeXt-Tiny.
    
    Args:
        config: Configuration dictionary (containing 'model' sub-dictionary).
        
    Returns:
        nn.Module: Instantiated PyTorch neural network.
    """
    model_cfg = config["model"]
    model_name = model_cfg["name"].lower()
    num_classes = model_cfg["num_classes"]
    
    # Check if pretrained is requested
    pretrained = model_cfg.get("pretrained", True)
    
    if model_name == "cnn_v2":
        return CustomCNNV2(num_classes=num_classes)
        
    elif model_name == "efficientnet_b0":
        return get_efficientnet_b0(num_classes=num_classes, pretrained=pretrained)
        
    elif model_name == "resnet50":
        return get_resnet50(num_classes=num_classes, pretrained=pretrained)
        
    elif model_name == "convnext_tiny":
        return get_convnext_tiny(num_classes=num_classes, pretrained=pretrained)
        
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")
