import torch
import numpy as np
import pandas as pd
from joblib import load, dump
import os
import numpy as np
import cv2
from PIL import Image
import PIL
from torchvision.transforms import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastprogress import master_bar, progress_bar
from sklearn.model_selection import train_test_split
import time
import pickle
import argparse
import random

class ERCTrainDataset(Dataset):
    def __init__(self, mels, labels, transforms, time_mask=0.1, freq_mask=0.1, spec_aug=True):
        super().__init__()
        self.mels = mels
        self.labels = labels
        self.transforms = transforms
        self.time_mask = time_mask
        self.freq_mask = freq_mask
        self.spec_aug = spec_aug

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        # crop 1sec
        image = self.mels[idx].copy()
        time_dim, base_dim = image.shape[1], image.shape[0]
        crop = np.random.randint(0, time_dim - base_dim)
        image = image[:, crop:crop + base_dim, ...]

        if self.spec_aug:
            freq_mask_begin = int(np.random.uniform(0, 1 - self.freq_mask) * base_dim)
            image[freq_mask_begin:freq_mask_begin + int(self.freq_mask * base_dim), ...] = 0
            time_mask_begin = int(np.random.uniform(0, 1 - self.time_mask) * base_dim)
            image[:, time_mask_begin:time_mask_begin + int(self.time_mask * base_dim), ...] = 0

        image = PIL.Image.fromarray(image[...,0], mode='L')
        image = self.transforms(image).div_(255)
        if self.labels is not None:
            label = np.asarray(self.labels[idx])
            label = torch.from_numpy(label).long()
        return (image, label) if self.labels is not None else image


class ERCTestDataset(Dataset):
    def __init__(self, mels, transforms, tta=5):
        super().__init__()
        self.mels = mels
        self.transforms = transforms
        self.tta = tta

    def __len__(self):
        return len(self.mels)  # * self.tta

    def __getitem__(self, idx):
        image = Image.fromarray(self.mels[idx][...,0], mode='L')
        time_dim, base_dim = image.size
        crop = np.random.randint(0, time_dim - base_dim)
        image = image.crop([crop, 0, crop + base_dim, base_dim])
        image = self.transforms(image).div_(255)
        return image