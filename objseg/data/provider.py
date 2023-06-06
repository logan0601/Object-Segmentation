import pickle
import os

import torch
import torch.nn as nn
from torch.utils import data
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from utils.typing import *


VERTEX_COLORS = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
]


def get_corners():
    """Get 8 corners of a cuboid. (The order follows OrientedBoundingBox in open3d)
        (y)
        2 -------- 7
       /|         /|
      5 -------- 4 .
      | |        | |
      . 0 -------- 1 (x)
      |/         |/
      3 -------- 6
      (z)
    """
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    return corners - [0.5, 0.5, 0.5]


def get_edges(corners):
    assert len(corners) == 8
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(corners[i] == corners[j]) == 2:
                edges.append((i, j))
    assert len(edges) == 12
    return edges


def load_pickle(filename) -> Dict[str, Any]:
    with open(filename, 'rb') as f:
        return pickle.load(f)


class FrustumDataset(data.Dataset):
    def __init__(
        self,
        data_dir: str,
        split_dir: str,
        split: str,
        process_dir: str,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.split_dir = split_dir
        self.split = split
        self.process_dir = process_dir
        # prefix
        with open(os.path.join(split_dir, f"{split}.txt"), 'r') as f:
            self.prefix = [line.strip() for line in f if line.strip()]

        # corners and edges
        self.corners = get_corners()
        self.edges = get_edges(self.corners)

    def __len__(self):
        return len(self.prefix)
    
    def __getitem__(self, index) -> Dict[str, Any]:
        prefix = self.prefix[index]

        # rgb, depth, meta
        rgb = np.array(Image.open(f"{self.data_dir}/{prefix}_color_kinect.png")) / 255
        depth = np.array(Image.open(f"{self.data_dir}/{prefix}_depth_kinect.png")) / 1000
        meta = load_pickle(f"{self.data_dir}/{prefix}_meta.pkl")
        intrinsic = meta["intrinsic"]

        data_dict = {
            "images": rgb,
            "depths": depth,
            "img_info": np.array([720, 1280]),
            "intrinsic": intrinsic
        }

        if self.split in ["train", "val"]:
            label = np.array(Image.open(f"{self.data_dir}/{prefix}_label_kinect.png"))
            bbox_data = np.load(f"{self.process_dir}/{prefix}_bbox.npz")
            gt_boxes = bbox_data["gt_boxes"]
            num_boxes = bbox_data["num_boxes"]

            data_dict.update({
                "labels": label,
                "gt_boxes": gt_boxes,
                "num_boxes": num_boxes
            })
            return data_dict
        
        elif self.split == "test":
            return data_dict

