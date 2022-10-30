import torch
import torch.nn as nn


class AlexNet(nn.Module):

    def __init__(self, num_classes):
        super(AlexNet, self).__init__()

        ## out_channels is the number of feature maps,
        # which is often equivalent to the number of kernels that you apply to the input

        self.c1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, padding=0, stride=4)
        self.c2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2, stride=1)
        self.c3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(6 * 6 * 256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.local_norm = nn.LocalResponseNorm(size=5)
        self.drop = nn.Dropout(.5)


    def forward(self, image):
        # batch Size , 3 , 227 , 227

        ## First Conv Layer ##
        net = self.c1(image)
        # 96 * 55 * 55
        net = self.relu(net)
        net = self.local_norm(net)
        net = self.max_pool(net)
        # 96 * 27 * 27

        ## Second Conv Layer ##
        net = self.relu(self.c2(net))
        # 256 * 27 * 27
        net = self.max_pool(self.local_norm(net))
        # 256 * 13 * 13

        ## Third Layer ##
        net = self.relu(self.c3(net))
        # 384 * 13 * 13

        ## Fourth Layer ##
        net = self.relu(self.c4(net))
        # 384 * 13 * 13

        ## Fifth ##
        net = self.relu(self.c5(net))
        # 256 * 13 * 13
        net = self.max_pool(net)
        # 256 * 6 * 6

        ## Fully Connected Layers
        # shape: [batch, 256 * 6 * 6]
        net = net.view(net.size(0), -1)
        net = self.drop(self.fc1(net))
        net = self.relu(net)

        net = self.drop(self.fc2(net))
        net = self.relu(net)

        output = self.fc3(net)

        return output



