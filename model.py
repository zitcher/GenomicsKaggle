from torch import nn

class Embeddor(nn.Module):
    def __init__(self, in_size=5, out_size=1024):
        """
            Simple CNN model to test data pipeline
        """
        super().__init__()

        self.encoder1 = nn.Linear(in_size, out_size // 2)
        self.encoder2 = nn.Linear(out_size // 2, out_size)

    def forward(self, input):
        output = self.encoder2(self.encoder1(input))
        return output
