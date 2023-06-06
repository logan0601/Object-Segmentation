import torch

from objseg.utils.typing import *


def generate_anchors(
    base_size: int, scales: Float[Tensor, "3"], ratios: Float[Tensor, "3"]
) -> Float[Tensor, "9 4"]:
    """ generate anchors """
    base_anchor = torch.as_tensor([[1, 1], [base_size, base_size]]).to(scales) - 1.0 # [2, 2]
    center = base_anchor.mean(dim=0)
    wh = base_anchor[1] - base_anchor[0] + 1.0

    # ratio
    size_ratios = (wh[0] * wh[1]) / ratios
    ws = torch.round(torch.sqrt(size_ratios))
    hs = torch.round(ws * ratios)

    # scale
    ws = ws[:, None] * scales[None, :]
    hs = hs[:, None] * scales[None, :]

    # anchor
    # FIXME: add 1
    wh = torch.stack([ws, hs], dim=-1).view(-1, 2)
    anchors = torch.cat([center - 0.5 * (wh - 1), center + 0.5 * (wh - 1)], dim=-1) + 1.0
    return anchors


def bbox_transform(
    anchors: Float[Tensor, "... N 4"], gt: Float[Tensor, "B M 4"]
) -> Float[Tensor, "B M 4"]:
    """ compute target transform from anchors to gt """
    if anchors.dim() == 2:
        anchors_w = anchors[:, 2] - anchors[:, 0] + 1.0
        anchors_h = anchors[:, 3] - anchors[:, 1] + 1.0
        anchors_x = anchors[:, 0] + 0.5 * anchors_w
        anchors_y = anchors[:, 1] + 0.5 * anchors_h

        gt_w = gt[..., 2] - gt[..., 0] + 1.0
        gt_h = gt[..., 3] - gt[..., 1] + 1.0
        gt_x = gt[..., 0] + 0.5 * gt_w
        gt_y = gt[..., 1] + 0.5 * gt_h

        dx = (gt_x - anchors_x[None]) / anchors_w
        dy = (gt_y - anchors_y[None]) / anchors_h
        dw = torch.log(gt_w / anchors_w[None])
        dh = torch.log(gt_h / anchors_h[None])
    
    elif anchors.dim() == 3:
        anchors_w = anchors[..., 2] - anchors[..., 0] + 1.0
        anchors_h = anchors[..., 3] - anchors[..., 1] + 1.0
        anchors_x = anchors[..., 0] + 0.5 * anchors_w
        anchors_y = anchors[..., 1] + 0.5 * anchors_h

        gt_w = gt[..., 2] - gt[..., 0] + 1.0
        gt_h = gt[..., 3] - gt[..., 1] + 1.0
        gt_x = gt[..., 0] + 0.5 * gt_w
        gt_y = gt[..., 1] + 0.5 * gt_h

        dx = (gt_x - anchors_x) / anchors_w
        dy = (gt_y - anchors_y) / anchors_h
        dw = torch.log(gt_w / anchors_w)
        dh = torch.log(gt_h / anchors_h)
    
    else:
        raise ValueError(f"Incorrect input shape of anchors of {anchors.shape}")

    target = torch.stack([dx, dy, dw, dh], 2)
    return target


def bbox_transform_inv(
    anchors: Float[Tensor, "B N 4"], deltas: Float[Tensor, "B N 4"],
) -> Float[Tensor, "B N 4"]:
    """ transform anchors according to deltas """
    w = anchors[..., 2] - anchors[..., 0] + 1.0
    h = anchors[..., 3] - anchors[..., 1] + 1.0
    x = anchors[..., 0] + 0.5 * w
    y = anchors[..., 1] + 0.5 * h

    dx, dy, dw, dh = deltas.chunk(chunks=4, dim=-1)

    pred_w = torch.exp(dw) * w[..., None]
    pred_h = torch.exp(dh) * h[..., None]
    pred_x = dx * w[..., None] + x[..., None]
    pred_y = dy * h[..., None] + y[..., None]
    
    pred_anchors = torch.cat(
        [
            pred_x - 0.5 * pred_w,
            pred_y - 0.5 * pred_h,
            pred_x + 0.5 * pred_w,
            pred_y + 0.5 * pred_h
        ],
        dim=-1
    )
    return pred_anchors


def bbox_overlaps(
    anchors: Float[Tensor, "N 4"], gt_boxes: Float[Tensor, "B K 4"]
) -> Float[Tensor, "B N K"]:
    """ compute overlap(iou) between all anchors and gt bbox """
    batch_size = gt_boxes.shape[0]

    N = anchors.shape[0]
    K = gt_boxes.shape[1]

    anchors = anchors[None].expand(batch_size, N, 4).contiguous()
    gt_boxes = gt_boxes[..., :4].contiguous()

    anchors_boxes_x = anchors[..., 2] - anchors[..., 0] + 1
    anchors_boxes_y = anchors[..., 3] - anchors[..., 1] + 1
    anchors_area = (anchors_boxes_x * anchors_boxes_y)[..., None]

    gt_boxes_x = gt_boxes[..., 2] - gt_boxes[..., 0] + 1
    gt_boxes_y = gt_boxes[..., 3] - gt_boxes[..., 1] + 1
    gt_boxes_area = (gt_boxes_x * gt_boxes_y)[:, None]

    # FIXME: height or width eq 1 mean zero area
    anchors_area_zero = (anchors_boxes_x == 1) | (anchors_boxes_y == 1)
    gt_area_zero = (gt_boxes_x == 1) | (gt_boxes_y == 1)

    boxes = anchors[:, :, None, :].expand(batch_size, N, K, 4)
    query_boxes = gt_boxes[:, None, :, :].expand(batch_size, N, K, 4)

    iw = (torch.minimum(boxes[..., 2], query_boxes[..., 2]) -
          torch.maximum(boxes[..., 0], query_boxes[..., 0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.minimum(boxes[..., 3], query_boxes[..., 3]) -
          torch.maximum(boxes[..., 1], query_boxes[..., 1]) + 1)
    ih[ih < 0] = 0

    union = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / union

    # anchors/gt_boxes cant be zero, overlaps with iou
    overlaps.masked_fill_(gt_area_zero[:, None].expand(batch_size, N, K), 0)
    overlaps.masked_fill_(anchors_area_zero[..., None].expand(batch_size, N, K), -1)
    return overlaps


def clip_boxes(
    boxes: Float[Tensor, "B N 4"],
    img_shape: Float[Tensor, "2"]
) -> Float[Tensor, "B N 4"]:
    """ clip out of image size boxes """
    h, w = img_shape
    boxes[..., 0].clamp_(0, w - 1)
    boxes[..., 1].clamp_(0, h - 1)
    boxes[..., 2].clamp_(0, w - 1)
    boxes[..., 3].clamp_(0, h - 1)
    return boxes
