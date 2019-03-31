import torch
import os
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
from utils import voc_label_indices, colormap2label, RandomCropTwoTensor

class FCN_load(data.Dataset):
    def __init__(self, train=True, size=(224, 224), num_class=21):
        self.size = size
        self.num_class = num_class
        txt_fname = 'train.txt' if train else 'trainval.txt'
        with open(txt_fname, 'r') as f:
            self.images = f.read().split()

    def __getitem__(self, index):
        data_path = 'JPEGImages/' + self.images[index] + '.jpg'
        label_path = 'SegmentationClass/' + self.images[index] + '.png'
        data = Image.open(data_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')
        to_tensor = transforms.ToTensor()
        data = to_tensor(data)
        label = to_tensor(label)
        print(data.shape, label.shape)
        label = voc_label_indices(label, colormap2label, num_class=self.num_class)

        return data, label

    def __len__(self):
        return len(self.images)

