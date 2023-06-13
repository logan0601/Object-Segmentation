import os
import json

import torch
import numpy as np
import cv2

from segment.utils.transform import get_corners, get_edges
from segment.utils.typing import *


edge = [[0, 1], [1, 2], [2, 3], [3, 0]]


def draw(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    for (i, j) in edge:
        cv2.line(
            image,
            (uv[i, 0], uv[i, 1]),
            (uv[j, 0], uv[j, 1]),
            color=(0, 1, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return image


def demo(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    image = (np.ascontiguousarray(image) * 255).astype(np.uint8)
    bbox = bbox.astype(np.int32)
    for i in range(bbox.shape[0]):
        x_lo, y_lo, x_hi, y_hi = bbox[i][:4]
        if x_lo == x_hi or y_lo == y_hi:
            break
        box = np.array([
            [x_lo, y_lo],
            [x_hi, y_lo],
            [x_hi, y_hi],
            [x_lo, y_hi]
        ])
        image = draw(image, box)

    return image


def demo_3d(
    image: np.ndarray,
    corner: np.ndarray,
    intrin: np.ndarray
) -> np.ndarray:
    image = np.ascontiguousarray(image).astype(np.uint8)
    corner = corner @ intrin.T
    uv = corner[:, :2] / corner[:, 2:]
    uv = uv.astype(np.int32)
    edges = get_edges(get_corners())
    for (i, j) in edges:
        cv2.line(
            image,
            (uv[i, 0], uv[i, 1]),
            (uv[j, 0], uv[j, 1]),
            color=(0, 1, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return image
