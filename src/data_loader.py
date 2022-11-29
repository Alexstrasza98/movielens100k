import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch


class Rating_Dataset(Dataset):
    def __init__(self, inputs, labels, ids):
        super().__init__()
        self.inputs = inputs
        self.labels = labels.values / 5
        self.ids = ids.values

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return (torch.from_numpy(self.inputs[item]).float(),
                torch.from_numpy(np.array([self.labels[item]])).float(),
                torch.from_numpy(np.array([self.ids[item]])))
