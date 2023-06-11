import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader

from segment.utils.typing import *


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


def get_corners() -> np.ndarray:
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


def get_edges(
    corners: Union[np.ndarray, List[int]]
) -> List[int]:
    assert len(corners) == 8
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(corners[i] == corners[j]) == 2:
                edges.append((i, j))
    assert len(edges) == 12
    return edges


def load_pickle(filename: str) -> Dict[str, Any]:
    with open(filename, 'rb') as f:
        return pickle.load(f)


class BaseDataModule:
    """ define base datamodule """
    batch_size: int
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset

    def general_loader(
        self,
        dataset: Dataset,
        batch_size: int
    ) -> DataLoader:
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError
    
    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError
    
    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError
