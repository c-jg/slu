import os

from torch.utils.data import Dataset
import pandas as pd

from utils import load_fbank


class FSCDataset(Dataset):
    def __init__(self, csv_path, label_map, base_dir):
        self.data = pd.read_csv(csv_path)
        self.label_map = label_map
        self.base_dir = base_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = os.path.join(self.base_dir, row['path'])

        x = load_fbank(path)  # shape (T, 320)

        label_str = f"{row['action']}-{row['object']}-{row['location']}"
        y = self.label_map[label_str]

        return x, y
