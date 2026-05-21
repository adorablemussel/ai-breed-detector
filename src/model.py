import torch.nn as nn
from torchvision import models

def get_model(class_count=120, pretrained=True):
    """
    Zoptymalizowana architektura: ResNet50 z regularyzacją Dropout.
    """
    # 1. Zmiana na potężniejszy ResNet50
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    # 2. Wymiana głowy (Head) 
    num_ftrs = model.fc.in_features
    
    # Dropout (50%) zapobiegający uczeniu się na pamięć
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5), 
        nn.Linear(num_ftrs, class_count)
    )

    return model