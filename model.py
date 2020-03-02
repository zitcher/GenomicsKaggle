import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Linear(nn.Module):
    def __init__(self, in_size=500, out_size=1):
        """
            Simple CNN model to test data pipeline
        """
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

        self.fc1 = nn.Linear(in_size, in_size)
        self.fc2 = nn.Linear(in_size, in_size)
        self.fc3 = nn.Linear(in_size, in_size)
        self.fc4 = nn.Linear(in_size, out_size)

    def forward(self, src):
        batch_size = src.size()[0]
        src = src.view(batch_size, -1)

        l1 = self.fc1(src)
        l2 = F.relu(self.fc2(l1))
        l3 = F.relu(self.fc3(l2))
        out = self.fc4(l3)

        return out
