import torch
import torch.nn as nn
import numpy as np


class Permute2d(nn.Module):
    def __init__(self, num_channels, shuffle):
        super().__init__()
        self.num_channels = num_channels
        print(num_channels)
        self.indices = np.arange(self.num_channels - 1, -1, -1).astype(np.long)
        self.indices_inverse = np.zeros((self.num_channels), dtype=np.long)
        print(self.indices_inverse.shape)
        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i
        if shuffle:
            self.reset_indices()

    def reset_indices(self):
        np.random.shuffle(self.indices)
        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i

    def forward(self, input, reverse=False):
        assert len(input.size()) == 3
        if not reverse:
            return input[:, :, self.indices]
        else:
            return input[:, :, self.indices_inverse]
