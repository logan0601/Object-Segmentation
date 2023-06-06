import torch
import torch.nn as nn

from objseg.utils.typing import *


def smooth_l1_loss(
    bbox_pred: Float[Tensor, "B 36 H W"],
    bbox_target: Float[Tensor, "B 36 H W"],
    bbox_inside_weight: Float[Tensor, "B 36 H W"],
    bbox_outside_weight: Float[Tensor, "B 36 H W"],
    sigma: float = 1.0,
    dims: List[int] = [1]
) -> Float[Tensor, "1"]:
    """ smooth l1 loss """
    sigma_2 = sigma ** 2
    bbox_diff = bbox_pred - bbox_target
    
    # inside loss
    in_box_diff = bbox_inside_weight * bbox_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1.0 / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2.0) * sigma_2 / 2.0 * smoothL1_sign \
                    + (abs_in_box_diff - (0.5 / sigma_2)) * (1.0 - smoothL1_sign)
    
    # outside loss
    out_loss_box = bbox_outside_weight * in_loss_box
    
    loss = out_loss_box
    loss = loss.sum(dim=dims)
    loss = loss.mean()
    return loss
