import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class NP1DCNN(nn.Module): # custom dialation, and no pooling
    def __init__(self, hidden_size=64, dilation=1, window=10):
        """
        :param hidden_size: int, size of hidden layers
        :param dilation: int, dilation value in the time dimension (1 for the other dimension, aka between the stocks)
        :param T: int, number of look back points
        """
        super(NP1DCNN, self).__init__()
        self.dilation = dilation
        self.hidden_size = hidden_size
        # First Layer
        # Input
        self.dilated_conv1 = nn.Conv2d(1, hidden_size, kernel_size=(1, 2), dilation=(1, self.dilation))
        self.relu1 = nn.ReLU()

        # Layer 2
        self.dilated_conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 2), dilation=(1, self.dilation))
        self.relu2 = nn.ReLU()

        # Layer 3
        self.dilated_conv3 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 2), dilation=(1, self.dilation))
        self.relu3 = nn.ReLU()

        # Layer 4
        self.dilated_conv4 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 2), dilation=(1, self.dilation))
        self.relu4 = nn.ReLU()

        # Output layer
        self.conv_final = nn.Conv2d(hidden_size, 1, kernel_size=(1, 2))

        self.window = window

    def forward(self, x):
        """

        :param x: Pytorch Variable, batch_size x 1 x n_stocks x T
        :return:
        """

        # First layer
        out = self.dilated_conv1(x)
        out = self.relu1(out)

        # Layer 2:
        out = self.dilated_conv2(out)
        out = self.relu2(out)

        # Layer 3:
        out = self.dilated_conv3(out)
        out = self.relu3(out)

        # Layer 4:
        out = self.dilated_conv4(out)
        out = self.relu4(out)

        # Final layer
        out = self.conv_final(out)
        out = out[:, :, :, -1]

        return out




class NP2DCNN(nn.Module): # custom dialation, and no pooling
    def __init__(self, hidden_size=64, dilation=1, window=10):
        """
        :param hidden_size: int, size of hidden layers
        :param dilation: int, dilation value in the time dimension (1 for the other dimension, aka between the stocks)
        :param T: int, number of look back points
        """
        super(NP2DCNN, self).__init__()
        self.dilation = dilation
        self.hidden_size = hidden_size
        # First Layer
        # Input
        self.dilated_conv1 = nn.Conv2d(2, hidden_size, kernel_size=(2, 2), dilation=(1, self.dilation))
        self.relu1 = nn.ReLU()

        # Layer 2
        self.dilated_conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(2, 2), dilation=(1, self.dilation))
        self.relu2 = nn.ReLU()

        # Layer 3
        self.dilated_conv3 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(2, 2), dilation=(1, self.dilation))
        self.relu3 = nn.ReLU()

        # Layer 4
        self.dilated_conv4 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(2, 2), dilation=(1, self.dilation))
        self.relu4 = nn.ReLU()

        # Output layer
        self.conv_final = nn.Conv2d(hidden_size, 1, kernel_size=(1, 2))

        self.window = window

    def forward(self, x):
        """

        :param x: Pytorch Variable, batch_size x 1 x n_stocks x T
        :return:
        """

        # First layer
        out = self.dilated_conv1(x)
        out = self.relu1(out)

        # Layer 2:
        out = self.dilated_conv2(out)
        out = self.relu2(out)

        # Layer 3:
        out = self.dilated_conv3(out)
        out = self.relu3(out)

        # Layer 4:
        out = self.dilated_conv4(out)
        out = self.relu4(out)

        # Final layer
        out = self.conv_final(out)
        out = out[:, :, :, -1]

        return out

