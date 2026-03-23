import torch
from torch.utils.data import Dataset

class TFRDataset(Dataset):
    def __init__(self, X, y, time_crop=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.time_crop = time_crop  # int или None

    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx]  # (C,F,T)
        if self.time_crop is not None:
            C, F, T = x.shape
            tc = self.time_crop
            if tc < T:
                t0 = torch.randint(0, T - tc + 1, (1,)).item()
                x = x[:, :, t0:t0+tc]
        return x, self.y[idx]
