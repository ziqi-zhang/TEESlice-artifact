from __future__ import print_function

import sys
import time

from torch import nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from python.logger_utils import Logger
from python.data import num_classes_dict, get_data
from python.common_net import register_weight_layer
from python.timer_utils import NamedTimerInstance, VerboseLevel

print("CUDA is available:", torch.cuda.is_available())
torch.backends.cudnn.deterministic = True

dtype = torch.float
device = torch.device("cuda")

minibatch = 128

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


dataset_name = "IDC"
data_path = "./data"
num_workers = 2
n_class = num_classes_dict[dataset_name]

loaders = get_data(dataset_name, data_path, minibatch, num_workers)
trainloader = loaders["train"]
testloader = loaders["test"]

class NetVgg16(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.norm1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.norm1_2 = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.norm2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.norm2_2 = nn.BatchNorm2d(128)
        self.NumElemFlatten = 8192

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.norm3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.norm3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.norm3_3 = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.norm4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.norm4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.norm4_3 = nn.BatchNorm2d(512)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.norm5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.norm5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.norm5_3 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.NumElemFlatten = 512

        self.normfc1 = nn.BatchNorm1d(512)
        self.normfc2 = nn.BatchNorm1d(512)
        self.normfc3 = nn.BatchNorm1d(10)
        self.fc1 = nn.Linear(self.NumElemFlatten, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, n_class)

    def forward_without_bn(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.pool(x)

        x = x.view(-1, self.NumElemFlatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_with_bn(self, x):
        x = F.relu(self.norm1_1(self.conv1_1(x)))
        x = F.relu(self.norm1_2(self.conv1_2(x)))
        x = self.pool(x)

        x = F.relu(self.norm2_1(self.conv2_1(x)))
        x = F.relu(self.norm2_2(self.conv2_2(x)))
        x = self.pool(x)

        x = F.relu(self.norm3_1(self.conv3_1(x)))
        x = F.relu(self.norm3_2(self.conv3_2(x)))
        x = F.relu(self.norm3_3(self.conv3_3(x)))
        x = self.pool(x)

        x = F.relu(self.norm4_1(self.conv4_1(x)))
        x = F.relu(self.norm4_2(self.conv4_2(x)))
        x = F.relu(self.norm4_3(self.conv4_3(x)))
        x = self.pool(x)

        x = F.relu(self.norm5_1(self.conv5_1(x)))
        x = F.relu(self.norm5_2(self.conv5_2(x)))
        x = F.relu(self.norm5_3(self.conv5_3(x)))
        x = self.pool(x)

        x = x.view(-1, self.NumElemFlatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    def forward(self, x):
        return self.forward_with_bn(x)


class Net(nn.Module):
    def __init__(self, n_class):
        super(Net, self).__init__()
        need_bias = False
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=need_bias)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, bias=need_bias)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1, bias=need_bias)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1, bias=need_bias)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1, bias=need_bias)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1, bias=need_bias)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv7 = nn.Conv2d(512, 512, 3, padding=1, bias=need_bias)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1, bias=need_bias)
        self.pool = nn.MaxPool2d(2, 2)
        # self.NumElemFlatten = int(262144 / minibatch)
        self.NumElemFlatten = 512
        self.fc1 = nn.Linear(self.NumElemFlatten, 512)
        self.fc2 = nn.Linear(512, 512, bias=need_bias)
        self.fc3 = nn.Linear(512, n_class, bias=need_bias)

        affine = True
        track_running_stats = True
        self.norm1 = nn.BatchNorm2d(64, affine=affine, track_running_stats=track_running_stats)
        self.norm2 = nn.BatchNorm2d(128, affine=affine, track_running_stats=track_running_stats)
        self.norm3 = nn.BatchNorm2d(256, affine=affine, track_running_stats=track_running_stats)
        self.norm4 = nn.BatchNorm2d(256, affine=affine, track_running_stats=track_running_stats)
        self.norm5 = nn.BatchNorm2d(512, affine=affine, track_running_stats=track_running_stats)
        self.norm6 = nn.BatchNorm2d(512, affine=affine, track_running_stats=track_running_stats)
        self.norm7 = nn.BatchNorm2d(512, affine=affine, track_running_stats=track_running_stats)
        self.norm8 = nn.BatchNorm2d(512, affine=affine, track_running_stats=track_running_stats)
        self.normfc1 = nn.BatchNorm1d(512, affine=affine, track_running_stats=track_running_stats)
        self.normfc2 = nn.BatchNorm1d(512, affine=affine, track_running_stats=track_running_stats)

    def forward_with_bn(self, x):
        x = self.pool(F.relu(self.norm1(self.conv1(x))))
        x = self.pool(F.relu(self.norm2(self.conv2(x))))
        x = self.pool(F.relu(self.norm4(self.conv4(F.relu(self.norm3(self.conv3(x)))))))
        x = self.pool(F.relu(self.norm6(self.conv6(F.relu(self.norm5(self.conv5(x)))))))
        x = self.pool(F.relu(self.norm8(self.conv8(F.relu(self.norm7(self.conv7(x)))))))
        x = x.view(-1, self.NumElemFlatten)
        x = F.relu(self.normfc1(self.fc1(x)))
        x = F.relu(self.normfc2(self.fc2(x)))
        x = self.fc3(x)
        return x

    def forward_without_bn(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.pool(F.relu(self.conv6(F.relu(self.conv5(x)))))
        x = self.pool(F.relu(self.conv8(F.relu(self.conv7(x)))))
        x = x.view(-1, self.NumElemFlatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_with_somebn(self, x):
        x = self.pool(self.norm1(F.relu(self.conv1(x))))
        x = self.pool(self.norm2(F.relu(self.conv2(x))))
        x = self.pool(self.norm3(F.relu(self.conv3(x))))
        x = self.pool(self.norm5(F.relu(self.conv5(x))))
        x = self.pool(self.norm7(F.relu(self.conv7(x))))
        x = x.view(-1, self.NumElemFlatten)
        x = self.normfc1(F.relu(self.fc1(x)))
        x = self.fc3(x)
        return x

    def forward(self, x):
        # return self.forward_without_bn(x)
        return self.forward_with_bn(x)
        # return self.forward_with_somebn(x)


# net = NetVgg16(10)
net = Net(n_class)
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

# register_weight_layer(net.conv1, "conv1")
# register_weight_layer(net.conv2, "conv2")
# register_weight_layer(net.conv3, "conv3")
# register_weight_layer(net.conv4, "conv4")
# register_weight_layer(net.conv5, "conv5")
# register_weight_layer(net.conv6, "conv6")
# register_weight_layer(net.conv7, "conv7")
# register_weight_layer(net.conv8, "conv8")
# register_linear_layer(net.fc1, "fc1")

sys.stdout = Logger()
print(f"Dataset: {dataset_name}")
NumShowInter = 10
NumEpoch = 200
# torch.set_num_threads(1)
training_start = time.time()
IterCounter = -1
running_loss = 0
# https://github.com/chengyangfu/pytorch-vgg-cifar10
for epoch in range(NumEpoch):  # loop over the dataset multiple times
    # running_loss = 0.0
    train_total, train_correct = 0, 0
    for i, data in enumerate(trainloader, 0):
        IterCounter += 1
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        # with NamedTimerInstance("TrainWithBatch", VerboseLevel.RUN):
        outputs = net(inputs)

        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if IterCounter % NumShowInter == NumShowInter - 1 or (epoch == 0 and i == 0):  # print every 2000 mini-batches
            elapsed_time = time.time() - training_start
            print('[%d, %5d, %6d, %6f] loss: %.3f, train acc: %5f' %
                  (epoch + 1, i + 1, IterCounter, elapsed_time, running_loss / NumShowInter, train_correct / train_total * 100))
            running_loss = 0.0
            correct = 0
            total = 0

            # store_layer_name = 'conv8'
            # store_name = f"{store_layer_name}_{epoch + 1}_{i + 1}"
            # store_layer(store_layer_name, store_name)

            with torch.no_grad():
                for test_data in testloader:
                    images, labels = test_data[0].to(device), test_data[1].to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                print('Accuracy of the network on the %d test images: %6f %%' % (total, 100 * correct / total))
    scheduler.step()

print('Finished Training')
