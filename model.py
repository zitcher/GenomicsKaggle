import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Linear(nn.Module):
    def __init__(self, in_size=500, out_size=1, dropout=0.5):
        """
            Simple CNN model to test data pipeline
        """
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

        self.fc1 = nn.Linear(in_size, in_size)
        self.fc2 = nn.Linear(in_size, in_size)
        self.fc3 = nn.Linear(in_size, in_size)
        self.fc4 = nn.Linear(in_size, in_size)
        self.fc5 = nn.Linear(in_size, in_size)
        self.fc6 = nn.Linear(in_size, out_size)

        # BatchNorm1d
        self.bn1 = nn.BatchNorm1d(in_size)
        self.bn2 = nn.BatchNorm1d(in_size)

        self.d = nn.Dropout(p=dropout)
        self.d1 = nn.Dropout(p=dropout)
        self.d2 = nn.Dropout(p=dropout)
        self.d3 = nn.Dropout(p=dropout)

    def forward(self, src):
        batch_size = src.size()[0]
        src = src.view(batch_size, -1)

        l1 = self.d1(F.elu(self.bn1(self.fc1(src))))

        l2 = F.relu(self.fc2(l1))
        l3 = self.d2(F.elu(self.bn2(self.fc3(l2))))
        l4 = F.relu(self.d3(self.fc4(l3)))
        l5 = F.relu(self.fc5(l4))
        out = F.relu(self.fc6(l5))

        return out.squeeze(1)
