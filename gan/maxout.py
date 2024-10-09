import torch
import torch.nn as nn


class MaxOut(nn.Module):
    def __init__(self, input_dim, num_pieces: int = 5):
        """
        Maxout layer

        Args:
            input_dim (int): Dimension of the input features.
            num_pieces (int): Number of linear pieces to compute max over.
        """
        super(MaxOut, self).__init__()
        self.input_dim = input_dim
        self.num_pieces = num_pieces

        self.fc = nn.Linear(
            in_features=input_dim, out_features=input_dim * num_pieces
        )

    def forward(self, x):
        """
        Forward pass of the Maxout layer

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Tensor of shape (batch_size, input_dim)
        """

        output = self.fc(x)
        output = output.view(output.size(0), self.input_dim, self.num_pieces)
        return torch.max(output, dim=2)[0]
