import torch
from FCN32s import FCN32s
from load import FCN_load
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from loss import Loss

batch_size = 2
num_workers = 0
momentem = 0.99
lr = 1e-4
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
epoch = 100

net = FCN32s()
net = net.to(device)
train_data = FCN_load()
val_data = FCN_load(train=False)

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=num_workers)
val_dataloader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers)
criterion = Loss
optimizer = optim.SGD(net.parameters(), lr, momentem)

for epoch in range(1, epoch + 1):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    print('epoch-' + str(epoch), ':')
    for i, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        y_hat = net(images)

        optimizer.zero_grad()
        loss = criterion(y_hat, labels)
        loss.backward()
        optimizer.step()

        train_l_sum += loss.item()
    print('epoch', epoch, 'loss', train_l_sum / (len(train_data) // batch_size))
    if epoch % 10 == 0:
        torch.save(net, 'epoch-' + epoch + '_saved_model.pth')




