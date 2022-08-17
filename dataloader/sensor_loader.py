import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import pathlib
import os
import glob

# data.shape = (N, 3, 5, 224, 224)

# hyperparameter
INPUT_SIZE = 512
FRAME = 5

# DataLoader
class CustomDataset(Dataset):
    def __init__(self, mode='Train'):
        super(CustomDataset, self).__init__()
        
        """
        :param folders: list of all the video folders
        :param frames: start frame, end frame and skip frame numpy array
        :param to_augment: boolean if the data is suppose to augment
        :param transform: transform function
        :param mode: train/test
        """

        self.data = []
        self.label = []
        self.mode = mode

        self.image_rgb = transforms.Compose([
            transforms.Resize([INPUT_SIZE, INPUT_SIZE]),
            transforms.ToTensor()
        ])
        #self.transform = transforms.ToTensor()

        for name in pathlib.Path(f'./sensor_img_b/{mode}/').glob('*.jpg'):
            category = os.path.splitext(name)[0].split('\\')[2]
  
            self.label.append(category)

            self.data.append((str(name), int(category[1])))
            self.data = sorted(self.data)



    def __len__(self):
        return len(self.data) // FRAME

    def __getitem__(self, idx):
        image_list = []
        
        for num in range(idx*FRAME, (idx+1)*FRAME):
            path = self.data[num][0]
            image = Image.open(f'./{path}')

            # (3,224,224)
            image = self.image_rgb(image)
            image_list.append(image)            
        
        # (3,224,224) -> (3,5,224,224)
        image_list = torch.stack(image_list, dim=1)

        label = torch.tensor(self.data[idx*FRAME][1])

        return image_list, label

    