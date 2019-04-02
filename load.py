import torch
import os
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.nn import functional as F
from utils import voc_label_indices, colormap2label

class FCN_load(data.Dataset):
    def __init__(self, train=True, num_class=21, tranform=None):
        self.num_class = num_class
        self.transform = tranform
        txt_fname = 'train.txt' if train else 'trainval.txt'
        with open(txt_fname, 'r') as f:
            self.images = f.read().split()

    def __getitem__(self, index):
        data_path = 'JPEGImages/' + self.images[index] + '.jpg'
        label_path = 'SegmentationClass/' + self.images[index] + '.png'
        data = Image.open(data_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')
        if self.transform is not None:
            data = self.transform(data)
            label = self.transform(label)
        else:
            data = transforms.CenterCrop(512)(data)
            label = transforms.CenterCrop(512)(label)
            data = transforms.ToTensor()(data)
            label = transforms.ToTensor()(label)
        # print(label.shape)
        label = voc_label_indices(label, colormap2label)

        return data, label

    def __len__(self):
        return len(self.images)
