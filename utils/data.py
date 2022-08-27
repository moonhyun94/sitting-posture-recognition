import os
import cv2
import pathlib
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from config import *


class CustomDataset(Dataset):

    def __init__(self, mode, augmentation=False):
        super(CustomDataset, self).__init__()

        self.augmentation = augmentation
        self.data_list = self._load_data_list(mode)

    def _load_data_list(self, mode):
        if mode == "train":
            data_path = os.path.join(FRAME_DATA_DIR, "train")
        elif mode == "test":
            data_path = os.path.join(FRAME_DATA_DIR, "test")
        else:
            raise KeyError(f"Invalid dataset mode: '{mode}'")

        data_list = []

        for person in pathlib.Path(data_path).glob("*"):
            if not person.is_dir():
                continue

            for cat in person.glob("*"):
                if not cat.is_dir():
                    continue

                for k in cat.glob("*"):
                    if not k.is_dir():
                        continue

                    video = []

                    for frame in k.glob("*.jpg"):
                        csv_path = os.path.join(SENSOR_DATA_DIR, f"{cat.name}_{person.name}_{k.name}.csv")
                        video.append((str(frame), csv_path, CAT_LIST.index(cat.name)))

                    data_list.append(video)

        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        video = self.data_list[index]

        if TIME_SPAN == 30:
            start_index = 0
        else:
            start_index = np.random.randint(0, 30 - TIME_SPAN)
            
        end_index = start_index + TIME_SPAN

        frames = []
        sensors = []

        # crop_h_range = [
        #     np.random.randint(0, HEIGHT//5),
        #     np.random.randint(HEIGHT - HEIGHT//5, HEIGHT)
        # ]
        # crop_w_range = [
        #     np.random.randint(0, WIDTH//5),
        #     np.random.randint(WIDTH - WIDTH//5, WIDTH)
        # ]

        for i in range(start_index, end_index):
            frame_path, csv_path, label = video[i]
            frame = self._load_frame(frame_path)
            sensor = self._load_sensor_data(csv_path, i)

            frames.append(frame)
            sensors.append(sensor)

        frames = np.stack(frames, axis=0)
        sensors = np.stack(sensors, axis=0)
        return frames, sensors, label

    def _load_frame(self, frame_path):
        frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if frame.shape[-1] == 4:
            frame = frame[..., :3]

        frame = cv2.resize(frame, dsize=(WIDTH, HEIGHT))
        if self.augmentation:
            crop_h_range = [
                np.random.randint(0, HEIGHT//5),
                np.random.randint(HEIGHT - HEIGHT//5, HEIGHT)
            ]
            crop_w_range = [
                np.random.randint(0, WIDTH//5),
                np.random.randint(WIDTH - WIDTH//5, WIDTH)
            ]
            frame = self._crop(frame, crop_h_range, crop_w_range)

        frame = (frame.astype(np.float32) - 128) / 128

        if self.augmentation and np.random.rand() < 0.5:
            frame = self._color_distortion(frame)
            frame = frame + np.random.normal(0, 0.01, size=frame.shape).astype(np.float32)

        frame = frame.transpose(2, 0, 1)
        return frame

    def _load_sensor_data(self, csv_path, index):
        csv = pd.read_csv(csv_path)
        csv_data = csv.iloc[:, -6:].values
        csv_data[:, :3] /= 100

        if self.augmentation:
            csv_data[:, :3] += np.random.normal(loc=0, scale=0.01, size=csv_data[:, :3].shape)
            csv_data[:, 3:] += np.random.normal(loc=0, scale=0.001, size=csv_data[:, 3:].shape)
        
        return csv_data[int((NUM_CSV_ROW/NUM_FRAMES) * index)].astype(np.float32)

    def _crop(self, frame, h_range, w_range):
        frame = frame[h_range[0]:h_range[1], w_range[0]:w_range[1]]
        frame = cv2.resize(frame, dsize=(WIDTH, HEIGHT))
        return frame

    def _color_distortion(self, frame):
        end_point = np.random.choice([-1, 1])

        if np.random.rand() < 0.5:
            frame[..., 0] = frame[..., 0] * 0.7 + end_point * 0.3
        if np.random.rand() < 0.5:
            frame[..., 1] = frame[..., 1] * 0.7 + end_point * 0.3
        if np.random.rand() < 0.5:
            frame[..., 2] = frame[..., 2] * 0.7 + end_point * 0.3

        return frame


# class SimCLRDataset(Dataset):

#     def __init__(self):
#         super(SimCLRDataset, self).__init__()

#         self.custom_dataset = CustomDataset("train", True)

#     def __len__(self):
#         return len(self.custom_dataset)

#     def __getitem__(self, index):

