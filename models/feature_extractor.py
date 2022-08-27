import torch
from torch import nn
from torchvision.models import inception_v3
from torchvision.models import Inception_V3_Weights


class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        inception_layers = [
            layer
            for name, layer in list(inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).named_children())[:-2]
            if name != "AuxLogits"
        ]

        self.features = nn.Sequential(
            *inception_layers
        )

    def forward(self, frames):
        features = self.features(frames)
        return features

