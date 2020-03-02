import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Transformer(nn.Module):
    def __init__(self, nin, ninp, nhead, nhid, nlayers, dropout=0.2, seq_len=100):
        """
            Simple CNN model to test data pipeline
        """
        super().__init__()
        self.ninp = ninp
        self.nhead = nhead
        self.nhid = nhid
        self.encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, nlayers)

        self.pos_encoder = nn.Linear(ninp, ninp)
        self.encoder = nn.Linear(nin, ninp)
        self.decoder = nn.Linear(seq_len * ninp, 1)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        batch_size = src.size()[0]
        src = self.encoder(src)
        mask = self._generate_square_subsequent_mask(len(src)).to(device)
        src = src + self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)
        output = output.view(batch_size, -1)
        output = self.decoder(output).squeeze(1)
        return output
