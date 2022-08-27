import torch
from torch import nn
from torchvision.models import inception_v3
from torchvision.models import Inception_V3_Weights


class SensorPredictor(nn.Module):

    def __init__(self, feature_extractor):
        super(SensorPredictor, self).__init__()

        self.features = feature_extractor

        self.accel_predictor = nn.Sequential(
            nn.Conv1d(2048, 2048, 3, stride=1, padding=0),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.1),
            
            nn.Conv1d(2048, 2048, 3, stride=1, padding=0),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.1),
            
            nn.Conv1d(2048, 2048, 3, stride=1, padding=0),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.1),
            
            nn.Conv1d(2048, 3, 4, stride=1, padding=0)
        )

        self.gyro_predictor = nn.Sequential(
            nn.Conv1d(2048, 2048, 3, stride=1, padding=0),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.1),
            
            nn.Conv1d(2048, 2048, 3, stride=1, padding=0),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.1),
            
            nn.Conv1d(2048, 2048, 3, stride=1, padding=0),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.1),
            
            nn.Conv1d(2048, 3, 4, stride=1, padding=0)
        )

    def forward(self, frames):
        N, T, C, H, W = frames.size()

        frames = frames.view(N*T, C, H, W)
        features = self.features(frames)
        features = features.view(N, T, -1)
        features = features.permute(0, 2, 1)

        accel_pred = self.accel_predictor(features)
        gyro_pred = self.gyro_predictor(features)

        sensor_pred = torch.cat([accel_pred, gyro_pred], dim=1)
        sensor_pred = sensor_pred.permute(0, 2, 1)
        
        return sensor_pred

