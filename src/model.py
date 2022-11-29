import torch
from torch import nn


class MovieRater_Simple(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim=1, model_name="default_model"):
        super().__init__()

        self.model_name = model_name

        self.dims = [input_dim] + hidden_units + [output_dim]
        self.layers = []
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1], bias=True))
            if i != len(self.dims) - 2:
                self.layers.append(nn.ReLU())
            else:
                self.layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)

        return x
