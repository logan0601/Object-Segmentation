import os

import torch
import numpy as np

from segment.utils.config import cfg
from segment.utils.typing import *


def backproject(
    dep: Float[Tensor, "H W"],
    intrin: Float[Tensor, "3 3"]
) -> Float[Tensor, "H W 3"]:
    """ back project image to space """
    v, u = np.indices(dep.shape)
    v = torch.from_numpy(v).to(dep)
    u = torch.from_numpy(u).to(dep)
    uv = torch.stack([
        u + 0.5, v + 0.5, torch.ones_like(dep)
    ], dim=-1)
    coords = uv @ torch.linalg.inv(intrin).T * dep[..., None]
    return coords


def normalize(
    pnt: Float[Tensor, "H W 3"],
    dim: int=1
) -> Float[Tensor, "H W 3"]:
    """ normalize along dim """
    pnt = pnt - pnt.mean(dim=dim)[:, None]
    size = (pnt.max(dim=dim)[0] - pnt.min(dim=dim)[0])[:, None].clamp(min=1e-5)
    pnt = pnt / size
    return pnt


def compute_intersection_over_union(
    pred: np.ndarray,
    gt: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    intersect = np.histogram(pred[pred == gt], bins=cfg.num_objects, range=[0, cfg.num_objects-1])[0]
    total_num_pred = np.histogram(pred, bins=cfg.num_objects, range=[0, cfg.num_objects-1])[0]
    total_num_gt = np.histogram(gt, bins=cfg.num_objects, range=[0, cfg.num_objects-1])[0]
    union = total_num_pred + total_num_gt - intersect
    return intersect, union, total_num_gt
