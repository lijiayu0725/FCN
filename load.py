from PIL import Image
from torch.utils.data import Dataset

from utils import *


class VOCSegDataset(Dataset):
    def __init__(self, crop_size, transforms, mode='train'):
        self.crop_size = crop_size
        self.transforms = transforms
        self.image_list, self.label_list = read_images(mode=mode)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        label = self.label_list[idx]
        image = Image.open(image)
        label = Image.open(label).convert('RGB')

        image, label = self.transforms(image, label, self.crop_size)
        a = label.numpy()
        return image, label

    def __len__(self):
        return len(self.image_list)
