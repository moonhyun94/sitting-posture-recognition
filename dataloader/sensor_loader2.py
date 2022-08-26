import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import pandas as pd
import numpy as np
from PIL import Image
import pathlib
import os
import glob

# data.shape = (N, 3, 5, 224, 224)

# hyperparameter
INPUT_SIZE = 512
# INPUT_SIZE = 224
FRAME = 5

# DataLoader
class CustomDataset(Dataset):
    def __init__(self, mode='Train'):
        super(CustomDataset, self).__init__()

        self.data = []
        self.label = []
        self.mode = mode

        self.image_rgb = transforms.Compose([
            transforms.Resize([INPUT_SIZE, INPUT_SIZE]),
            transforms.ToTensor()
        ])
        for name in pathlib.Path(f'/data/moon/datasets/sitting-posture-recognition/sensor_data/{mode}/').glob('*.jpg'):
            category = os.path.splitext(name)[0].split('/')[-1].split('_')[0][1]
            # category = os.path.splitext(name)[0].split('/')[2]
            self.label.append(category)
            # print('data path', name)
            # print('label', category)
            # self.data.append((str(name), int(category[1])))
            self.data.append((str(name), int(category)))
            self.data = sorted(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx][0]
        image = Image.open(path)
            # (3,224,224)
        image = self.image_rgb(image)
        # print(np.array(image).shape)

        # print(image.shape)
        label = torch.tensor(self.data[idx][1])
        # print('path', path)
        # print('label', label)

        return image, label

    