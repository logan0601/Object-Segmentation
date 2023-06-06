import pickle
import os

import torch
import torch.nn as nn
from torch.utils import data
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from utils.typing import *


def load_pickle(filename) -> Dict[str, Any]:
    with open(filename, 'rb') as f:
        return pickle.load(f)


class FrustumDataset(data.Dataset):
    def __init__(
        self, data_dir: str, split_dir: str, split: str
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.split_dir = split_dir
        self.split = split
        # prefix
        with open(os.path.join(split_dir, f"{split}.txt"), 'r') as f:
            self.prefix = [os.path.join(data_dir, line.strip()) for line in f if line.strip()]

    def __len__(self):
        return len(self.prefix)
    
    def __getitem__(self, index) -> Dict[str, Any]:
        prefix = self.prefix[index]

        # rgb, depth, meta
        rgb = np.array(Image.open(f"{prefix}_color_kinect.png")) / 255
        depth = np.array(Image.open(f"{prefix}_depth_kinect.png")) / 1000
        meta = load_pickle(f"{prefix}_meta.pkl")

        if self.split in ["train", "val"]:
            return {
                "rgb": rgb,
                "depth": depth,
                "intrinsic": meta["intrinsic"]
            }
        
        elif self.split == "test":
            return {
                "rgb": rgb,
                "depth": depth,
                "intrinsic": meta["intrinsic"]
            }

