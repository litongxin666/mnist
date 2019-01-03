import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import torchvision
import numpy
from torch.autograd import Variable
import torchvision.datasets.mnist as mnist
from skimage import io

transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

data_train = datasets.MNIST(root = "/home/litongxin/mnist/data",
                            transform=transform,
                            train = True,
                            download = True)

data_test = datasets.MNIST(root="/home/litongxin/mnist/data",
                           transform = transform,
                           train = False)

root="./mnist/data/raw"

train_set = (
    mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
)

test_set = (
    mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
)



def convert_to_img(train=True):
  if(train):
    f = open(root + 'train.txt', 'w')
    data_path = root + '/train/'
    if(not os.path.exists(data_path)):
      os.makedirs(data_path)
    for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
      img_path = data_path + str(i) + '.jpg'
      io.imsave(img_path, img.numpy())
      int_label = str(label).replace('tensor(', '')
      int_label = int_label.replace(')', '')
      f.write(img_path + ' ' + str(int_label) + '\n')
    f.close()
  else:
    f = open(root + 'test.txt', 'w')
    data_path = root + '/test/'
    if (not os.path.exists(data_path)):
      os.makedirs(data_path)
    for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
      img_path = data_path + str(i) + '.jpg'
      io.imsave(img_path, img.numpy())
      int_label = str(label).replace('tensor(', '')
      int_label = int_label.replace(')', '')
      f.write(img_path + ' ' + str(int_label) + '\n')
    f.close()

convert_to_img(True)
convert_to_img(False)

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size = 64,
                                               shuffle = True)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size = 64,
                                               shuffle = True)


class Model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.layer1=nn.Sequential(nn.Conv2d(1,16,kernel_size=3),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2,stride=2))
        self.layer3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc=nn.Sequential(nn.Linear(128*4*4,1024),
                              nn.ReLU(inplace=True),
                              nn.Linear(1024,128),
                              nn.ReLU(inplace=True),
                              nn.Linear(128,10))
        def forward(self,x):
            x=self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x=x.view(x.size(0),-1)
            x=self.fc(x)
            return x

model = Model()
print(model)
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
n_epochs = 5
model.load_state_dict(torch.load('model_parameter.pkl'))

for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0
    print("Epoch {}/{}".format(epoch, n_epochs))
    print("-" * 10)
    for data in data_loader_train:
        X_train, y_train = data
        X_train, y_train = Variable(X_train), Variable(y_train)
        outputs = model(X_train)
        _, pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = cost(outputs, y_train)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        running_correct += torch.sum(pred == y_train.data)
    testing_correct = 0
    for data in data_loader_test:
        X_test, y_test = data
        X_test, y_test = Variable(X_test), Variable(y_test)
        outputs = model(X_test)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == y_test.data)
    print "Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(running_loss / len(data_train),
                                                                                      100 * running_correct/len(data_train),
                                                                                      100 * testing_correct/len(data_test))

torch.save(model.state_dict(), "model_parameter.pkl")



data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                          batch_size = 4,
                                          shuffle = True)
X_test, y_test = next(iter(data_loader_test))
inputs = Variable(X_test)
pred = model(inputs)
_,pred = torch.max(pred, 1)

print("Predict Label is:", [ i for i in pred.data])
print("Real Label is:",[i for i in y_test])

img = torchvision.utils.make_grid(X_test)
#img = img.numpy().transpose(1,2,0)

#std = [0.5,0.5,0.5]
#mean = [0.5,0.5,0.5]
#img = img*std+mean
plt.imshow(img)


