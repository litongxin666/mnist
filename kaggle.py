import os
import pandas as pd
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
import torch as t
from tqdm import tqdm
from torch.autograd import Variable
from mnist_models import conv_net,to_image,fc_net,AlexNet
import signal


class conv_net(t.nn.Module):
    def __init__(self):
        super(conv_net,self).__init__()
        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(1,10,5,1,1),
            t.nn.MaxPool2d(2),
            t.nn.ReLU(),
            t.nn.BatchNorm2d(10)
        )
        self.conv2 = t.nn.Sequential(
            t.nn.Conv2d(10,20,5,1,1),
            t.nn.MaxPool2d(2),
            t.nn.ReLU(),
            t.nn.BatchNorm2d(20) # num_features为通道数
        )
        self.fc1 = t.nn.Sequential(
            t.nn.Linear(500,60),
            t.nn.Dropout(0.5),
            t.nn.ReLU()
        )
        self.fc2 = t.nn.Sequential(
            t.nn.Linear(60,20),
            t.nn.Dropout(0.5),
            t.nn.ReLU()
        )
        self.fc3 = t.nn.Linear(20,10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1,500)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


TYPE = 'cla'
METHOD = 'conv'
EPOCHS = 400
BATCH_SIZE = 500
LR = 0.001

train = pd.read_csv('/home/litongxin/data/train.csv')
data = train.drop('label',axis=1)
test = pd.read_csv('/home/litongxin/data/test.csv')
test_data = t.from_numpy(test.values).float()
data = data.values #delete label_x

y = train['label'].values
y = t.from_numpy(y).long() #label convert to tensor
data = t.from_numpy(data).float() #data convert to tensor
data,y = Variable(data),Variable(y)

data = to_image(data)
test_data = to_image(test_data)
net = conv_net()

criterion = t.nn.CrossEntropyLoss()
optim = t.optim.Adam(net.parameters(),lr=0.001,weight_decay=0.0)


for epoch in tqdm(range(EPOCHS)):
    index = 0
    if epoch % 100 == 0:
        for param_group in optim.param_groups:
            LR = LR * 0.9
            param_group['lr'] = LR
    for i in tqdm(range(int(len(data)/BATCH_SIZE)),total=int(len(data)/BATCH_SIZE)):

        batch_x = data[index:index + BATCH_SIZE]
        batch_y = y[index:index + BATCH_SIZE]
        prediction = net.forward(batch_x)
        loss = criterion(prediction, batch_y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        index += BATCH_SIZE  # 进入下一个batch
        # if loss <= 0.3:
            # losses.append(loss)
        # plt.plot(losses)
        # plt.pause(0.001)

        print(loss)
t.save(net.state_dict(),'/home/litongxin/data/cnn.pth')
# plt.ioff()
submission = pd.read_csv("/home/litongxin/data/sample_submission.csv")

print "=======Predicting========"

net.eval()

test_data = Variable(test_data)

result = t.Tensor()

index = 0

for i in tqdm(range(int(test_data.shape[0]/BATCH_SIZE)),total=int(test_data.shape[0]/BATCH_SIZE)):
    label_prediction = net(test_data[index:index+BATCH_SIZE])
    index += BATCH_SIZE
    result = t.cat((result,label_prediction),0)

_,submission['Label'] = t.max(result.data,1)
submission.to_csv("submission.csv",index=False)




