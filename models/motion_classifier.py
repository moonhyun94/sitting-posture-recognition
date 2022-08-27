import torch
from torch import nn
from config import *


class MotionClassifier(nn.Module):

    def __init__(self, feature_extractor):
        super(MotionClassifier, self).__init__()

        self.features = feature_extractor

        self.sensor_features = nn.Sequential(
            nn.Conv1d(6, 128, 5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            
            nn.Conv1d(128, 128, 5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),

            nn.Conv1d(128, 128, 5, stride=1, padding=2),
        )

        self.motion_classifier = nn.Sequential(
            nn.Conv1d(2048 + 128, len(CAT_LIST), TIME_SPAN, stride=1, padding=0),
            # nn.Conv1d(128, len(CAT_LIST), TIME_SPAN, stride=1, padding=0),
        )

    def forward(self, frames, sensors):
        N, T, C, H, W = frames.size()

        frames = frames.view(N*T, C, H, W)
        frames_features = self.features(frames)
        frames_features = frames_features.view(N, T, -1)
        frames_features = frames_features.permute(0, 2, 1)

        sensors = sensors.permute(0, 2, 1)
        sensors_features = self.sensor_features(sensors)

        features = torch.cat([frames_features, sensors_features], dim=1)
        motion_pred_logits = self.motion_classifier(features)
        # motion_pred_logits = self.motion_classifier(sensors_features)
        motion_pred_logits = motion_pred_logits.squeeze(-1)
        
        return motion_pred_logits

