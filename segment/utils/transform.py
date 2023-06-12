import json
import pickle
import numpy as np

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


def estimate_poses(
    center: np.ndarray,
    size: np.ndarray,
    rotation: np.ndarray,
    extrinsic: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    corners = get_corners()
    corners = corners * size
    corners = corners @ rotation.T + center
    corners = corners @ extrinsic[:3, :3].T + extrinsic[:3, 3]

    model_mat = np.zeros([4, 4], dtype=np.float32)
    model_mat[:3, :3] = rotation
    model_mat[:3, 3] = center
    model_mat[3, 3] = 1
    pose_mat = extrinsic @ model_mat
    rot = pose_mat[:3, :3]
    assert np.linalg.norm(np.identity(3) - rot @ rot.T) < 1e-5
    return corners, pose_mat
