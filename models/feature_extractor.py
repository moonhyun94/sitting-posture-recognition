import torch
from torch import nn
from torchvision.models import inception_v3
from torchvision.models import Inception_V3_Weights
from models.blocks import *


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


class ConvMixerFeatureExtractor(nn.Module):
    
    def __init__(self):
        super(ConvMixerFeatureExtractor, self).__init__()

        self.features = nn.Sequential(
            PatchEmbedding(3, 256, 8),

            ConvMixerBlock(256, 7),
            ConvMixerBlock(256, 7),
            ConvMixerBlock(256, 7),
            ConvMixerBlock(256, 7),
            ConvMixerBlock(256, 7),

            nn.Conv2d(256, 256, 16, stride=1, padding=0)
        )

    def forward(self, x):
        return self.features(x)

