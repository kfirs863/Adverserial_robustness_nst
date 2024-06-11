import torch
from torch import nn
from torchvision import models


class StyleSelectionCNN(nn.Module):
    def __init__(self, num_target_classes=10):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_extractor.eval()

        # use the pretrained model to classify num_target_classes
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x