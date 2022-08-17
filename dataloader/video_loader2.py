import os
from sklearn.model_selection import train_test_split

import torch
from torchvision import transforms
import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import random


class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(self, dataset='video', split='train', clip_len=30, preprocess=False):
        # original video data path, preprocessed video data path
        self.root_dir, self.output_dir = '/data/moon/datasets/sitting-posture-recognition/dataset', '/data/moon/datasets/sitting-posture-recognition/dataset'
        folder = os.path.join('/data/moon/datasets/sitting-posture-recognition/dataset', split) # 'train', 'test'
        self.clip_len = clip_len
        self.split = split
        
        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 112
        self.resize_width = 112
        self.crop_size = 95
        
        self.transform = transforms.RandomResizedCrop((self.resize_height, self.resize_width), scale=(0.8, 1.0))

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, labels = [], []

        for img in sorted(os.listdir(folder)):
            label = img.split('_')[0]
            self.fnames.append(os.path.join(folder, img)) # /data/moon/datasets/sitting-posture-recognition/dataset/train/마이쭈/A1
            labels.append(label)

        self.fnames = sorted(self.fnames)
        self.fnames = [self.fnames[i:i+30] for i in range(0,len(self.fnames), 30)]
        
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)
        self.label_array = [np.squeeze(np.unique(self.label_array[i:i+30])) for i in range(0, len(self.label_array), 30)]

        
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # print(index)
        # Loading and preprocessing.
        buffer = self.load_frames(self.fnames[index]) # 30, 112, 112, 3
        labels = np.array(self.label_array[index]) # (,)
        
        # if self.split == 'test':
        #     print('index', index)
        #     # print('filename', self.fnames[index][0])
        #     # print('labels', labels)
        
        if self.split == 'train':
            if np.random.rand() < 0.5:
                buffer = self.gaussian_noise(buffer)
                
            buffer = self.normalize(buffer)
            buffer = self.to_tensor(buffer)
            buffer = torch.from_numpy(buffer)
            
            if np.random.rand() < 0.5:
                buffer = self.crop(buffer)
        else:
            buffer = self.normalize(buffer)
            buffer = self.to_tensor(buffer)
            buffer = torch.from_numpy(buffer)
        # print(self.fnames[index])
        # print(self.label_array[index])
        return buffer, torch.from_numpy(labels)


    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer


    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, files):

        frame_count = len(files)
        
        buffer = np.empty((30, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        
        for i, frames in enumerate(files):
            frame = cv2.imread(frames)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.array(frame).astype(np.float64)
            frame = cv2.resize(frame, (self.resize_width, self.resize_height))
            buffer[i] = frame

        return buffer
    
    def crop(self, buffer):
        # print('do crop')
        seed = np.random.randint(1000)
        
        torch.random.manual_seed(seed)
        random.seed(seed)
        
        for i in range(buffer.shape[1]):
            buffer[:, i] = self.transform(buffer[:, i])
        return buffer
        
    def gaussian_noise(self, buffer):
        for i, frame in enumerate(buffer):
            noise = np.random.normal(0.0, 5.0, size=(buffer[i].shape[0],buffer[i].shape[1],buffer[i].shape[2]))
            buffer[i] = noise + frame
        return buffer