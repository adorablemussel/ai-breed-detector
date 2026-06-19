import torch.nn as nn
from torchvision import models

def get_model(class_count=120, pretrained=True):
    """
    Zoptymalizowana architektura: ConvNeXt-Tiny z regularyzacją Dropout.
    """
    
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = models.convnext_tiny(weights=weights)

    num_ftrs = model.classifier[2].in_features
    
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.5), 
        nn.Linear(num_ftrs, class_count)
    )

    return model