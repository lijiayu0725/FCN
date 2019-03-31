import torch
from FCN32s import FCN32s
from load import FCN_load
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

batch_size = 2
num_workers = 0
momentem = 0.99
lr = 0.001
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

model = FCN32s()
model = model.to(device)
train_data = FCN_load()
val_data = FCN_load(train=False)

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=num_workers)
val_dataloader = DataLoader(val_data)

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentem)
criterion = nn.BCELoss().to(device)

for i, (data, label) in enumerate(train_dataloader):
    data = data.to(device)
    label = label.to(device)
    y_pred = model(data)
    print(y_pred.shape, label.shape)
    loss = criterion(y_pred, label)
    print(i, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
