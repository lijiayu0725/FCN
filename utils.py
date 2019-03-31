import torch
from torchvision import transforms
import numpy as np

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

colormap2label = np.zeros(256 ** 3)
for i, colormap in enumerate(VOC_COLORMAP):
    colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i

def voc_label_indices(colormap, colormap2label, num_class):
    to_pil = transforms.ToPILImage()
    colormap = to_pil(colormap)
    colormap = np.array(colormap).astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    label = colormap2label[idx]
    one_hot = np.zeros((num_class, label.shape[0], label.shape[1]))
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            one_hot[int(label[i][j]), i, j] = 1

    return torch.tensor(one_hot).float()

def RandomCropTwoTensor(image1, image2, size):
    h, w = image1.shape[-2:]
    new_h, new_w = size
    # if new_h >= h:
    #     image1 = torch.cat((image1, torch.zeros((image1.shape[0], new_h - h, image1.shape[2]))), dim=1)
    #     image2 = torch.cat((image2, torch.zeros((image2.shape[0], new_h - h, image2.shape[2]))))
    #     new_h = h
    #     print(image1.shape)
    # if new_w >= w:
    #     image1 = torch.cat((image1, torch.zeros((image1.shape[0], image1.shape[1], new_w - w))), dim=2)
    #     image2 = torch.cat((image2, torch.zeros((image2.shape[0], image2.shape[1], new_w - w))))
    #     new_w = w
    print(image1.shape, image2.shape)
    top = np.random.randint(0, h - new_h) if new_h != h else 0
    left = np.random.randint(0, w - new_w) if new_w != w else 0
    image1 = image1[:, top: top + new_h, left: left + new_w]
    image2 = image2[:, top: top + new_h, left: left + new_w]

    return image1, image2