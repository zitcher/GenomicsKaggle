import torch
from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):
    def __init__(
        self,
        channels,
        width,
        height,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        num_kernels=3,
        kernel_size=(4, 20),
        output_size=2,
        pool_size=(4, 4)
    ):
        """
            CNN model
        """
        super().__init__()
        # TODO: initialize the hyperparameters
        self.channels = channels
        self.output_size = output_size
        self.channels = channels
        self.height = height
        self.width = width
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=num_kernels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode='zeros'
        )

        self.pool = torch.nn.MaxPool2d(
            kernel_size=pool_size,
            padding=padding,
            dilation=dilation,
            stride=pool_size
        )

        # conv uout
        cheight, cwidth = self.calc_out_conv2d(
            (height, width), padding=padding, dilations=dilation, kernels=kernel_size, stride=stride)

        # pool out
        pheight, pwidth = self.calc_out_pool2d(
            (cheight, cwidth), padding=padding, dilations=dilation, kernels=pool_size, stride=pool_size)

        # print("height width", height, width)
        # print("cheight cwidth", num_kernels, cheight, cwidth)
        # print("pheight pwidth", num_kernels, pheight, pwidth)
        self.fc = nn.Linear(num_kernels * pheight * pwidth, output_size)

    def calc_out_conv2d(self, dims, padding, dilations, kernels, stride):
        out = [0] * len(dims)
        for i in range(len(dims)):
            out[i] = dims[i] + 2 * padding[i] - \
                dilations[i] * (kernels[i] - 1) - 1
            out[i] = int(out[i] / stride[i]) + 1
        return tuple(out)

    def calc_out_pool2d(self, dims, padding, dilations, kernels, stride):
        out = [0] * len(dims)
        for i in range(len(dims)):
            out[i] = dims[i] + 2 * padding[i] - \
                dilations[i] * (kernels[i] - 1) - 1
            out[i] = int(out[i] / stride[i]) + 1
        return tuple(out)

    def forward(self, input):
        # TODO: write forward propagation
        cout = self.conv(input)
        # print(cout.size())

        pout = self.pool(F.relu(cout))
        # print(pout.size())

        out = self.fc(torch.flatten(pout))

        return out
