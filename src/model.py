import torch
from torch import nn
from src.configs import *
import pdb


class MovieRater_Simple(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim=1, model_name="simple"):
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


class MovieRater_Embedding(MovieRater_Simple):
    def __init__(self, input_dim, hidden_units, feature_tags, output_dim=1, model_name="embedding"):
        self.feature_tags = feature_tags
        self.embedding_layer = {}
        self.final_input_dim = 0
        for name, cols in feature_tags.items():
            if name != "numeric":
                self.embedding_layer[name] = nn.Linear(len(cols), EMBEDDING_SIZE, bias=False)
                self.final_input_dim += EMBEDDING_SIZE
            else:
                self.final_input_dim += len(cols)

        super(MovieRater_Embedding, self).__init__(self.final_input_dim, hidden_units, output_dim, model_name)
        self.embedding_layer = nn.ModuleDict(self.embedding_layer)

    def forward(self, x):
        embeddings = []
        for name, cols in self.feature_tags.items():
            if name == "numeric":
                embeddings.append(x[..., cols])
            else:
                embeddings.append(self.embedding_layer[name](x[..., cols]))
        embeddings = torch.concat(embeddings, dim=1)

        return super(MovieRater_Embedding, self).forward(embeddings)

