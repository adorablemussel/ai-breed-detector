import torch.nn as nn
from torchvision import models

def get_model(class_count=120, pretrained=True):
    """
    Zoptymalizowana architektura: ConvNeXt-Tiny z regularyzacją Dropout.
    """
    # 1. Zmiana z ResNet50 na nowoczesny ConvNeXt-Tiny
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = models.convnext_tiny(weights=weights)

    # 2. Wymiana głowy (Head) dedykowanej dla struktury ConvNeXt
    # classifier to sekcja nn.Sequential, gdzie ostatnia warstwa liniowa ma indeks [2]
    num_ftrs = model.classifier[2].in_features
    
    # Budujemy nowy klasyfikator z Dropout (50%) zapobiegającym uczeniu się na pamięć
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.5), 
        nn.Linear(num_ftrs, class_count)
    )

    return model