from torch import optim
from torch.utils.data import DataLoader

from FCN8s import fcn8s
from load import VOCSegDataset
from loss import Loss
from utils import *

num_epochs = 80
train_batch = 1
val_batch = 16
learning_rate = 1e-2
weight_decay = 1e-4
num_classes = 21
momentum = 0.99
device = 'cuda' if torch.cuda.is_available() else 'cpu'

input_shape = (512, 512)
voc_train = VOCSegDataset(input_shape, image_transform)
voc_val = VOCSegDataset(input_shape, image_transform, mode='val')
train_data = DataLoader(voc_train, train_batch, shuffle=True)
val_data = DataLoader(voc_val, val_batch)

net = fcn8s(num_classes=num_classes).to(device)
criterion = Loss
optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

for epoch in range(num_epochs):
    print('epoch-%d: ' % epoch)
    train_loss = 0
    train_acc = 0
    train_acc_cls = 0
    train_mean_iu = 0
    train_fwavacc = 0
    net = net.train()
    for i, (image, label) in enumerate(train_data):
        image = image.to(device)

        label = label.to(device)
        y_hat = net(image)
        loss = criterion(y_hat, label)
        print('\tbatch-%d, loss: %f' % (i, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        label_pred = torch.argmax(y_hat, dim=1).data.cpu().numpy()
        label_true = label.data.cpu().numpy()

        for t, p in zip(label_true, label_pred):
            acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(t, p, num_classes)
            train_acc += acc
            train_acc_cls += acc_cls
            train_mean_iu += mean_iu
            train_fwavacc += fwavacc
    torch.save(net, 'saved_models/epoch-%d_saved_model.pth' % epoch)
    net = net.eval()
    eval_loss = 0
    eval_acc = 0
    eval_acc_cls = 0
    eval_mean_iu = 0
    eval_fwavacc = 0
    print('evaluating...')
    for i, (image, label) in enumerate(val_data):
        with torch.no_grad():
            image = image.to(device)
            label = label.to(device)
            # forward
            y_hat = net(image)
            loss = criterion(y_hat, label)
            eval_loss += loss.item()

            label_pred = torch.argmax(y_hat, dim=1).data.cpu().numpy()
            label_true = label.data.cpu().numpy()
            for t, p in zip(label_true, label_pred):
                acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(t, p, num_classes)
                eval_acc += acc
                eval_acc_cls += acc_cls
                eval_mean_iu += mean_iu
                eval_fwavacc += fwavacc

    epoch_str = ('Epoch: {}, Train Loss: {:.5f}, Train Acc: {:.5f}, Train Mean IU: {:.5f}, \
    Valid Loss: {:.5f}, Valid Acc: {:.5f}, Valid Mean IU: {:.5f} '.format(
        epoch, train_loss / len(train_data), train_acc / len(voc_train), train_mean_iu / len(voc_train),
               eval_loss / len(val_data), eval_acc / len(voc_val), eval_mean_iu / len(voc_val)))
    print(epoch_str)

print('---finished!---')
