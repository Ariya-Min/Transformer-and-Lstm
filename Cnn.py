import pandas as pd
import numpy as np
import torch
from torch import nn, Tensor
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size=20
num_epochs = 100
learning_rate = 0.001
#data process
data_csv=pd.read_csv("industry_ret_data_1500.csv")
dataset = data_csv.values
dataset=np.delete(dataset,0,axis=1)
dataset = dataset.astype('float32')
dataset=standardization(dataset)
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)
data_X, data_Y = create_dataset(dataset,input_size )
train_X = data_X[:1000]
train_Y = data_Y[:1000]
val_X=data_X[1000:1200]
val_Y=data_Y[1000:1200]
test_X = data_X[1200:]
test_Y = data_Y[1200:]
train_X = train_X.reshape(-1, 1,433, input_size )
train_Y = train_Y.reshape(-1,433)
test_X = test_X.reshape(-1, 1,433, input_size )
test_Y=test_Y.reshape(-1,433)
val_X = val_X.reshape(-1, 1,433, input_size )
val_Y = val_Y.reshape(-1, 433)
train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)
test_x = torch.from_numpy(test_X)
test_y=torch.from_numpy(test_Y)
val_x=torch.from_numpy(val_X)
val_y = torch.from_numpy(val_Y)
#print(train_y.shape,train_x.shape)
class Cnn(nn.Module):
    def __init__(self, out_node):
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1),  # 卷积层
            # nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=1),  # 最大池化层
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1),
            # nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(5978, 1000),  # 线性层
            #nn.ReLU(inplace=True),  # relu激活函数
            nn.Linear(1000, out_node),
            nn.Dropout(),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):  #
        out = self.conv(x)
        #print(out.shape)
        out = self.fc(out)
       # print(out.shape)
        return out

model=Cnn(433).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for i in range(num_epochs):
    output=model(train_x.to(device))
    loss=criterion(output,train_y.to(device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    val_out=model(val_x.to(device))
    val_loss=criterion(val_out,val_y.to(device))
    print('Epoch: {}, Loss: {:.5f},val_Loss: {:.5f}'.format(i+1, loss,val_loss))

test_out=model(test_x.to(device))
test_loss=criterion(test_out,test_y.to(device))
t_var=np.var(test_Y)
print('test_Loss: {:.5f}'.format(test_loss))
print("var:",t_var)


