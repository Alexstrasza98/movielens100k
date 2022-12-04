import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os

from src.configs import TITLE_FEATURE_PATH


class Rating_Dataset(Dataset):
    def __init__(self, inputs, labels, ids, with_title=False):
        super().__init__()
        self.inputs = inputs
        self.labels = labels.values / 5
        self.ids = ids.values
        self.with_title = with_title

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        input_sample = torch.from_numpy(self.inputs[item]).float()
        label_sample = torch.from_numpy(np.array([self.labels[item]])).float()
        id_sample = torch.from_numpy(np.array([self.ids[item]]))

        if self.with_title:
            title = torch.load(os.path.join(TITLE_FEATURE_PATH, f"{id_sample[0]}.pt"))
            input_sample = torch.concat([input_sample, title[0]])

        return input_sample, label_sample, id_sample
