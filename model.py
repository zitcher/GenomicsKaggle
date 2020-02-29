import torch
from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):
    def __init__(
        self,
        channels,
        width,
        height,
        batch_size,
        conv_structure,
        stride=(1, 1),
        dilation=(1, 1),
        groups=1,
        num_kernels=3,
        output_size=2,
    ):
        """
            Simple CNN model to test data pipeline
        """
        super().__init__()
        self.batch_size = batch_size
        self.channels = channels
        self.output_size = output_size
        self.channels = channels
        self.height = height
        self.width = width
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.num_kernels = num_kernels
        self.conv_structure = conv_structure


        self.drop_layer = nn.Dropout(p=0.5)


        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=num_kernels,
            kernel_size=conv_structure[0][0],
            padding=conv_structure[0][1],
            stride=stride,
            dilation=dilation,
            groups=groups,
            padding_mode='zeros'
        )


        self.pool1 = torch.nn.MaxPool2d(
            kernel_size=conv_structure[1][0],
            padding=conv_structure[1][1],
            dilation=(dilation),
            stride=conv_structure[1][2]
        )

        self.norm1 = torch.nn.BatchNorm2d(num_kernels)

        n_out_channels = num_kernels * num_kernels
        self.conv2 = nn.Conv2d(
            in_channels=num_kernels,
            out_channels=n_out_channels,
            kernel_size=conv_structure[2][0],
            padding=conv_structure[2][1],
            stride=stride,
            dilation=dilation,
            groups=groups,
            padding_mode='zeros'
        )


        self.pool2 = torch.nn.MaxPool2d(
            kernel_size=conv_structure[3][0],
            padding=conv_structure[3][1],
            dilation=dilation,
            stride=conv_structure[3][2],
        )

        self.norm2 = torch.nn.BatchNorm2d(n_out_channels)

        # conv uout
        c1height, c1width = self.calc_out_conv2d(
            (height, width), padding=conv_structure[0][1], dilations=dilation, kernels=conv_structure[0][0], stride=stride)
        print("height width", height, width)
        # pool out
        p1height, p1width = self.calc_out_pool2d(
            (c1height, c1width), padding=conv_structure[1][1], dilations=dilation, kernels=conv_structure[1][0], stride=conv_structure[1][2])
        print("p1height p1width", p1height, p1width)

        c2height, c2width = self.calc_out_conv2d(
            (p1height, p1width), padding=conv_structure[2][1], dilations=dilation, kernels=conv_structure[2][0], stride=stride)
        print("c2height c2width", c2height, c2width)
        # pool out
        p2height, p2width = self.calc_out_pool2d(
            (c2height, c2width), padding=conv_structure[3][1], dilations=dilation, kernels=conv_structure[3][0], stride=conv_structure[3][2])
        print("n_out_channels p2height p2width", n_out_channels, p2height, p2width)
        self.fc1 = nn.Linear(n_out_channels * p2height * p2width, n_out_channels * p2height * p2width // 2)
        self.fc2 = nn.Linear(n_out_channels * p2height * p2width // 2, output_size)

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
        # print("input", input.size())

        c1out = self.drop_layer(F.relu(self.norm1(self.conv1(input))))
        # print("cout", cout.size())

        p1out = self.pool1(c1out)
        # print("pout", pout.size())

        c2out = self.drop_layer(F.relu(self.norm2(self.conv2(p1out))))
        p2out = self.pool2(c2out)

        fc1out = self.drop_layer(F.relu(self.fc1(p2out.view(input.size()[0], -1))))
        out = self.fc2(fc1out)

        return out
