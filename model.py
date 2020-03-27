from torch import nn

class Embeddor(nn.Module):
    def __init__(self, num_embeddings=5, embedding_dim=10):
        """
            Simple CNN model to test data pipeline
        """
        super().__init__()

        self.encoder = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, input):
        output = self.encoder(input)
        return output
