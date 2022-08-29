import torch
from torch import nn
from config import *


class MotionClassifier(nn.Module):

    def __init__(self, feature_extractor):
        super(MotionClassifier, self).__init__()

        self.features = feature_extractor

        self.embedder = nn.Sequential(
            nn.Conv1d(2048, 2048, 1, stride=1, padding=0),
            nn.LeakyReLU(0.1),
            
            nn.Conv1d(2048, 2048, 1, stride=1, padding=0),
        )

    def forward(self, frames, sensors):
        N, T, C, H, W = frames.size()

        frames = frames.view(N*T, C, H, W)
        frames_features = self.features(frames)
        frames_features = frames_features.view(N, T, -1)
        
        return motion_pred_logits

