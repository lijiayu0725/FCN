import random

from PIL import Image
from matplotlib import pyplot as plt

from load import VOCSegDataset
from utils import *

num_images = 6
epoch_used = 8

net = torch.load('saved_models/epoch-%d_saved_model.pth' % epoch_used)
voc_test = VOCSegDataset((512, 512), image_transform, mode='test')
cm = np.array(colormap).astype('uint8')


def predict(image, label):  # 预测结果
    image = image.unsqueeze(0).cuda()
    out = net(image)
    pred = torch.argmax(out, dim=1).squeeze().cpu().data.numpy()
    pred = cm[pred]
    a = label.numpy()
    return pred, cm[label.numpy().astype(np.int)]


_, figs = plt.subplots(num_images, 3, figsize=(12, 10))
for i in range(num_images):
    n = random.randint(0, len(voc_test.image_list))
    image, label = voc_test[n]
    pred, label = predict(image, label)
    figs[i, 0].imshow(Image.open(voc_test.image_list[n]))
    figs[i, 0].axes.get_xaxis().set_visible(False)
    figs[i, 0].axes.get_yaxis().set_visible(False)
    figs[i, 1].imshow(label)
    figs[i, 1].axes.get_xaxis().set_visible(False)
    figs[i, 1].axes.get_yaxis().set_visible(False)
    figs[i, 2].imshow(pred)
    figs[i, 2].axes.get_xaxis().set_visible(False)
    figs[i, 2].axes.get_yaxis().set_visible(False)

plt.show()
