import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
#data_csv = pd.read_csv('C:\\Users\\86134\\Desktop\\temp.csv', header=None,usecols=[1])
from torch.utils.data import Dataset
import os
from torch.autograd import Variable

# class CustomDataset(Dataset):
#     def __init__(self,csv_dir, transform=None, target_transform=None):
#         self.csv = pd.read_csv(csv_dir,usecols=[1])
#         self.label = self.csv.iloc[:,1]
#         self.csv_dir = csv_dir
#         self.transform = transform
#         self.target_transform = target_transform
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, idx):
#         return idx,self.csv.iloc[idx,1]
# def create_dataset(dataset, look_back=2):
#     dataX, dataY = [], []
#     for i in range(len(dataset) - look_back):
#         a = dataset[i:(i + look_back)]
#         dataX.append(a)
#         dataY.append(dataset[i + look_back])
#     return [np.array(dataX), np.array(dataY)]
#
# training_data = CustomDataset(
#     csv_dir="C:\\Users\\dengchun\\Desktop\\temp.csv",
#     transform=create_dataset()[0].reshape(-1, 1, 2),
#     target_transform= create_dataset()[1].reshape(-1, 1, 1)
# )
#
# # Download test data from open datasets.
# test_data = CustomDataset(
#     csv_dir="C:\\Users\\dengchun\\Desktop\\temp.csv",
#     transform=create_dataset()[0].reshape(-1, 1, 2),
#     target_transform=create_dataset()[1].reshape(-1, 1, 1)
# )
#
# batch_size = 1
#
# train_dataloader = DataLoader(training_data, batch_size=batch_size)
# test_dataloader = DataLoader(test_data, batch_size=batch_size)
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")
#
#
# class lstm_reg(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
#         super(lstm_reg, self).__init__()
#
#         self.rnn = nn.LSTM(input_size, hidden_size, num_layers)  # rnn
#         self.reg = nn.Linear(hidden_size, output_size)  # ??????
#
#     def forward(self, x):
#         x, _ = self.rnn(x)  # (seq, batch, hidden)
#         s, b, h = x.shape
#         x = x.view(s * b, h)  # ?????????????????????????????????
#         x = self.reg(x)
#         x = x.view(s, b, -1)
#         return x
#
# net = lstm_reg(2, 4).to(device)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
# def train(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     model.train()
#     for batch, (X, y) in enumerate(dataloader):
#         X, y = X.to(device), y.to(device)
#
#         # Compute prediction error
#         pred = model(X)
#         loss = loss_fn(pred, y)
#
#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if batch % 100 == 0:
#             loss, current = loss.item(), batch * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
#
# def test(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval()
#     test_loss, correct = 0, 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
#
# epochs = 5
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_dataloader, net, criterion, optimizer)
#     test(test_dataloader, net, criterion)
# print("Done!")
data_csv=pd.read_csv("industry_ret_data_1500.csv")
dataset = data_csv.values
dataset=np.delete(dataset,0,axis=1)
dataset = dataset.astype('float32')
max_value = np.max(dataset,axis=0)
min_value = np.min(dataset,axis=0)
scalar = max_value - min_value
dataset = list((map(lambda x: x / scalar, dataset)))


def create_dataset(dataset, look_back=4):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

data_X, data_Y = create_dataset(dataset)

train_X = data_X[:1000]
train_Y = data_Y[:1000]
val_X=data_X[1000:1200]
val_Y=data_Y[1000:1200]
test_X = data_X[1200:]
test_Y = data_Y[1200:]

train_X = train_X.reshape(-1, 433, 4)
train_Y = train_Y.reshape(-1, 433, 1)
test_X = test_X.reshape(-1, 433, 4)
test_Y=test_Y.reshape(-1,433,1)
val_X = val_X.reshape(-1, 433, 4)
val_Y = val_Y.reshape(-1, 433, 1)

train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)
test_x = torch.from_numpy(test_X)
test_y=torch.from_numpy(test_Y)
val_x=torch.from_numpy(val_X)
val_y = torch.from_numpy(val_Y)
print(train_x.shape,train_y.shape)

class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(lstm_reg, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)  # rnn
        self.reg = nn.Linear(hidden_size, output_size)  # ??????

    def forward(self, x):
        x, _ = self.rnn(x)  # (seq, batch, hidden)
        s, b, h = x.shape
        x = x.view(s * b, h)  # ?????????????????????????????????
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x

net = lstm_reg(4,4)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

# ????????????
tr_loss=[]
val_loss=[]
for e in range(100):
    var_x = Variable(train_x)
    var_y = Variable(train_y)
    valx=Variable(val_x)
    valy=Variable(val_y)

    # ????????????
    out = net(var_x)
    val_out=net(valx)
    loss = criterion(out, var_y)
    v_loss=criterion(val_out, valy)
    tr_loss.append(loss)
    val_loss.append(v_loss)
    # ????????????
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Epoch: {}, Loss: {:.5f},val_Loss: {:.5f}'.format(e + 1, loss.data,v_loss))

net = net.eval() # ?????????????????????

test_data = Variable(test_x)
test_tar = Variable(test_y)
pred_test = net(test_x) # ????????????????????????
test_loss=criterion(pred_test,test_tar)
print('ave_val_Loss: {:.5f},test_loss:{:.5f}'.format(sum(val_loss)/len(val_loss),test_loss) )
# ?????????????????????
pred_test = pred_test.view(-1).data.numpy()
a=np.zeros(4)
pre=np.r_[a,pred_test]

