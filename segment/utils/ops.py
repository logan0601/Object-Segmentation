import os

import torch
import numpy as np

from transforms3d.euler import euler2mat

from segment.utils.config import cfg
from segment.utils.typing import *


def scale_tensor(
    dat: Num[Tensor, "... D"],
    inp_scale: Tuple[float, float],
    tgt_scale: Tuple[float, float],
) -> Num[Tensor, "... D"]:
    if inp_scale is None:
        inp_scale = (0, 1)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


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
    dim: int = 1
) -> Float[Tensor, "H W 3"]:
    """ normalize along dim """
    pnt = pnt - pnt.mean(dim=dim, keepdim=True)
    size = (pnt.max(dim=dim, keepdim=True)[0] \
            - pnt.min(dim=dim, keepdim=True)[0]).clamp(min=1e-5)
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


def generate_bbox(
    pc: Float[Tensor, "M 3"]
) -> Float[Tensor, "8 3"]:
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
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    zmin, zmax = z.min(), z.max()
    bbox = torch.as_tensor([
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmin, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmin]
    ])
    return bbox


def euler_to_rotation(
    angle: Float[Tensor, "3"]
) -> Float[Tensor, "3 3"]:
    ai, aj, ak = angle

    si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
    ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    i, j, k = 0, 1, 2

    M = torch.eye(3).to(angle)
    M[i, i] = cj
    M[i, j] = sj*si
    M[i, k] = sj*ci
    M[j, i] = sj*sk
    M[j, j] = -cj*ss+cc
    M[j, k] = -cj*cs-sc
    M[k, i] = -sj*ck
    M[k, j] = cj*sc+cs
    M[k, k] = cj*cc-ss
    return M


def eval_rdiff_with_sym_axis(pred_rotation, gt_rotation, sym_axis):
    """
    Compute rotation error (unit: degree) based on symmetry axis.
    """
    if sym_axis == "x":
        x1, x2 = pred_rotation[:, 0], gt_rotation[:, 0]
        diff = np.sum(x1 * x2)
    elif sym_axis == "y":
        y1, y2 = pred_rotation[:, 1], gt_rotation[:, 1]
        diff = np.sum(y1 * y2)
    elif sym_axis == "z":
        z1, z2 = pred_rotation[:, 2], gt_rotation[:, 2]
        diff = np.sum(z1 * z2)
    else:  # sym_axis == "", i.e. no symmetry axis
        mat_diff = np.matmul(pred_rotation, gt_rotation.T)
        diff = mat_diff.trace()
        diff = (diff - 1) / 2.0

    diff = np.clip(diff, a_min=-1.0, a_max=1.0)
    return np.arccos(diff) / np.pi * 180  # degree


def eval_rdiff(pred_rotation, gt_rotation, geometric_symmetry):
    """
    Compute rotation error (unit: degree) based on geometric symmetry.
    """
    syms = geometric_symmetry.split("|")
    sym_axis = ""
    sym_N = np.array([1, 1, 1])  # x, y, z
    for sym in syms:
        if sym.find("inf") != -1:
            sym_axis += sym[0]
        elif sym != "no":
            idx = ord(sym[0]) - ord('x')
            value = int(sym[1:])
            sym_N[idx] = value
    if len(sym_axis) >= 2:
        return 0.0
    
    assert sym_N.min() >= 1

    gt_rotations = []
    for xi in range(sym_N[0]):
        for yi in range(sym_N[1]):
            for zi in range(sym_N[2]):
                R = euler2mat(
                    2 * np.pi / sym_N[0] * xi,
                    2 * np.pi / sym_N[1] * yi,
                    2 * np.pi / sym_N[2] * zi,
                )
                gt_rotations.append(gt_rotation @ R)

    r_diffs = []
    for gt_rotation in gt_rotations:
        r_diff = eval_rdiff_with_sym_axis(pred_rotation, gt_rotation, sym_axis)
        r_diffs.append(r_diff)
    
    r_diffs = np.array(r_diffs)
    return r_diffs.min()


def eval_tdiff(pred_translation, gt_translation):
    """
    Compute translation error (unit: cm).
    """
    t_diff = pred_translation - gt_translation
    return np.linalg.norm(t_diff, ord=2) * 100


def eval_pose(pred_pose, gt_pose, geometric_symmetry="zinf|x2"):
    r_diff = eval_rdiff(pred_pose[:3, :3], gt_pose[:3, :3], geometric_symmetry)
    t_diff = eval_tdiff(pred_pose[:3, 3], gt_pose[:3, 3])
    return r_diff, t_diff
