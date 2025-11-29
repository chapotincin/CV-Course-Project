import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()

        if mode == 1:
            # 423x423 rgb image
            # first convolution layer input 3 channel, output 40, kernel size 5, stride 1
            self.conv1 = nn.Conv2d(3, 40, 3, padding=1)
            # second convolution layer input 40, output 40, kernel size 5, stride 1
            self.conv2 = nn.Conv2d(40, 40, 3, padding=1)
            # third convolution layer input 40, output 40, kernel size 5, stride 1
            #self.conv3 = nn.Conv2d(40, 40, 3, 2,padding=1)
            # fully connected hidden layer with 100 neurons and input of 17x17 image with 40 channels
            self.L1 = nn.Linear(40 * 17 * 17, 100)
            # output layer with 10 output neurons
            self.L2 = nn.Linear(100, 2)

            self.forward = self.model_1


        else:
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)

    def model_1(self, X):
        # apply ReLU function to input for non linearity after convolution
        X = torch.relu(self.conv1(X))
        # max pooling layer with kernel size 2 and stride 1 to reduce spatial dimensionality
        X = F.max_pool2d(X, 2, 2)
        # apply ReLU function to input for non linearity after convolution
        X = torch.relu(self.conv2(X))
        # max pooling layer with kernel size 2 and stride 1 to reduce spatial dimensionality
        X = F.max_pool2d(X, 2, 2)
        # apply ReLU function to input for non linearity after convolution
        # X = torch.relu(self.conv3(X))
        # # max pooling layer with kernel size 2 and stride 1 to reduce spatial dimensionality
        # X = F.max_pool2d(X, 2, 2)
        # flatten image to shape (batch_size, 40*17*17)
        X = X.view(-1, 40 * 17 * 17)
        # apply ReLU function to input for high level representation
        X = torch.relu(self.L1(X))
        # drouput layer with probability 0.5 to reduce overfitting
        X = F.dropout(X, p=0.5, training=self.training)
        # apply ReLU function to input for high level representation
        # X = torch.relu(self.L2(X))
        # #drouput layer with probability 0.5 to reduce overfitting
        # X=F.dropout(X, p=0.5,training=self.training)
        # #perform softmax on output layer
        return self.L2(X)

