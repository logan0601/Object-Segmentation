import torch
import torch.nn as nn
import numpy as np

from objseg.utils.transform import generate_anchors, clip_boxes
from objseg.utils.transform import bbox_transform, bbox_transform_inv, bbox_overlaps
from objseg.utils.typing import *


ValidScale = Union[Tuple[float, float], Num[Tensor, "2 D"]]


def scale_tensor(
    dat: Num[Tensor, "... D"], inp_scale: ValidScale, tgt_scale: ValidScale
):
    if inp_scale is None:
        inp_scale = (0, 1)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    if isinstance(tgt_scale, Tensor):
        assert dat.shape[-1] == tgt_scale.shape[-1]
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


def unmap(
    data: Float[Tensor, "B ..."],
    count: int,
    inds: Int[Tensor, "N"],
    batch_size: int,
    fill: float = 0.0
) -> Float[Tensor, "B ..."]:
    """ unmap subset data back to origin """
    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.shape[2]).fill_(fill).type_as(data)
        ret[:, inds, :] = data
    return ret


def nms(
    boxes: Float[Tensor, "N 4"],
    scores: Float[Tensor, "N"],
    thresh: float
) -> Int[Tensor, "M"]:
    """ non-maximum suppression """
    # https://github.com/amusi/Non-Maximum-Suppression/blob/master/nms.py
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size() > 0:
        i = order[0]
        keep.append(i.item())

        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        w = torch.maximum(0.0, xx2 - xx1 + 1)
        h = torch.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = torch.where(ovr <= thresh)[0]
        order = order[inds + 1]
    
    return torch.as_tensor(keep).long()


class ProposalLayer(nn.Module):
    """ output proposals by apply transformations to boxes """
    def __init__(
        self,
        feat_stride: int,
        scales: Float[Tensor, "3"],
        ratios: Float[Tensor, "3"]
    ) -> None:
        self.feat_stride = feat_stride
        self.anchors = generate_anchors(feat_stride, scales, ratios)
        self.num_anchors = self.anchors.shape[0]

    def forward(
        self,
        scores: Float[Tensor, "B 18 H W"],
        bbox_deltas: Float[Tensor, "B 36 H W"],
        img_info: Float[Tensor, "2"],
        state: str,
    ) -> Float[Tensor, "B M 5"]:
        # first num_anchors are bg probs, second num_anchors are fg probs
        scores = scores[:, self.num_anchors:]

        # train/val configuration
        if state in ["train", "val"]:
            (
                pre_nms_topN,
                post_nms_topN,
                nms_thresh
            ) = (
                12000,
                2000,
                0.7
            )
        elif state == "test":
            (
                pre_nms_topN,
                post_nms_topN,
                nms_thresh
            ) = (
                6000,
                300,
                0.7
            )

        batch_size = bbox_deltas.shape[0]
        feat_height, feat_width = scores.shape[2:4]

        # generate sliding window centers
        shift_x = torch.arange(feat_width).to(scores) * self.feat_stride
        shift_y = torch.arange(feat_height).to(scores) * self.feat_stride
        shift_x, shift_y = torch.meshgrid(shift_x, shift_y, indexing="xy")
        shifts = torch.stack(
            [
                shift_x.view(-1),
                shift_y.view(-1),
                shift_x.view(-1),
                shift_y.view(-1)
            ],
            dim=-1
        )
        shifts = shifts.contiguous().type_as(scores).float()

        A = self.num_anchors
        K = shifts.shape[0]

        # generate anchors
        anchors = self.anchors.type_as(scores)
        anchors = anchors[None] + shifts[:, None]
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4)

        # transpose scores and bbox to the same size as anchors
        scores = scores.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)

        # convert anchors to proposals
        proposals = bbox_transform_inv(anchors, bbox_deltas)
        proposals = clip_boxes(proposals, img_info)

        # apply NMS
        order = torch.sort(scores, 1, descending=True)
        output = scores.new_zeros(batch_size, post_nms_topN, 5)
        for i in range(batch_size):
            # remove proposals with either height/width < threshold
            proposals_single = proposals[i]
            scores_single = scores[i]
            order_single = order[i]

            if pre_nms_topN > 0 and pre_nms_topN < scores_single.numel():
                order_single = order_single[:pre_nms_topN]
            
            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single]

            # apply nms
            keep_idx_i = nms(proposals_single, scores_single, nms_thresh)
            keep_idx_i = keep_idx_i[:post_nms_topN]

            proposals_single = proposals_single[keep_idx_i]
            scores_single = scores_single[keep_idx_i]

            n_proposals = proposals_single.shape[0]
            output[i, :, 0] = i
            output[i, :n_proposals, 1:] = proposals_single
        
        # only operation, no gradient
        output = output.detach()

        # return rois of [B M 5] [index, bbox]
        return output


class AnchorTargetLayer(nn.Module):
    """ supervised anchor loss """
    def __init__(
        self,
        feat_stride: int,
        scales: Float[Tensor, "3"],
        ratios: Float[Tensor, "3"]
    ) -> None:
        super().__init__()

        self.feat_stride = feat_stride
        self.scales = scales
        self.anchors = generate_anchors(feat_stride, scales, ratios)
        self.num_anchors = self.anchors.shape[0]

        self.positive_overlap_thresh = 0.7
        self.negative_overlap_thresh = 0.3

        self.rpn_fg_frac = 0.5
        self.rpn_batch_size = 256

        self.rpn_bbox_inside_weight = 1.0

        # allow boxes to sit over the edge by a small amount
        self.allowed_border = 0  # default is 0

    def forward(
        self,
        scores: Float[Tensor, "B 18 H W"],
        gt_boxes: Optional[Float[Tensor, "B M 5"]],
        img_info: Float[Tensor, "B 2"],
    ) -> Dict[str, Any]:
        batch_size = gt_boxes.shape[0]
        feat_height, feat_width = scores.shape[2:4]

        # generate sliding window centers
        shift_x = torch.arange(feat_width).to(scores) * self.feat_stride
        shift_y = torch.arange(feat_height).to(scores) * self.feat_stride
        shift_x, shift_y = torch.meshgrid(shift_x, shift_y, indexing="xy")
        shifts = torch.stack(
            [
                shift_x.view(-1),
                shift_y.view(-1),
                shift_x.view(-1),
                shift_y.view(-1)
            ],
            dim=-1
        )
        shifts = shifts.contiguous().type_as(scores).float()

        A = self.num_anchors
        K = shifts.shape[0]

        anchors = self.anchors.type_as(gt_boxes)
        all_anchors = anchors[None] + shifts[:, None]
        all_anchors = all_anchors.view(K * A, 4)

        # remove out-of-image anchors
        keep = ((all_anchors[:, 0] >= -self.allowed_border) &
                (all_anchors[:, 1] >= -self.allowed_border) &
                (all_anchors[:, 2] < img_info[0, 1] + self.allowed_border) &
                (all_anchors[:, 3] < img_info[0, 0] + self.allowed_border)
            )
        inside_ids = torch.nonzero(keep).view(-1)
        anchors = all_anchors[inside_ids]

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = gt_boxes.new_full((batch_size, inside_ids.shape[0]), fill_value=-1)
        bbox_inside_weights = gt_boxes.new_zeros((batch_size, inside_ids.shape[0]))
        bbox_outside_weights = gt_boxes.new_zeros((batch_size, inside_ids.shape[0]))

        # compute overlap between anchors and gt_bbox [B, N, M]
        overlaps = bbox_overlaps(anchors, gt_boxes)

        # max_overlaps: most related gt_box across each inside anchor [B, N]
        # gt_max_overlaps: most related anchor across each gt_box [B, M]
        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
        gt_max_overlaps, _ = torch.max(overlaps, 1)

        # negative label: dont related to every gt_box
        labels[max_overlaps < self.negative_overlap_thresh] = 0

        # positive label: at least one anchor intersect for each gt_boxes, this anchor is positive
        gt_max_overlaps[gt_max_overlaps == 0] = 1e-5
        keep = torch.sum(overlaps.eq(gt_max_overlaps[:, None].expand_as(overlaps)), 2)
        if torch.sum(keep) > 0:
            labels[keep > 0] = 1

        # positive label: those whose iou larger than positive thresh
        labels[max_overlaps >= self.positive_overlap_thresh] = 1

        num_fg = int(self.rpn_fg_frac * self.rpn_batch_size)
        sum_fg = torch.sum((labels == 1).int(), 1)
        sum_bg = torch.sum((labels == 0).int(), 1)
        for i in range(batch_size):
            # subsample positive labels if we have too many
            if sum_fg[i] > num_fg:
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.shape[0])).type_as(gt_boxes).long()
                disable_inds = fg_inds[rand_num[:fg_inds.shape[0]-num_fg]]
                labels[i][disable_inds] = -1

            num_bg = self.rpn_batch_size - torch.sum((labels == 1).int(), 1)[i]

            # subsample negative labels if we have too many
            if sum_bg[i] > num_bg:
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                rand_num = torch.from_numpy(np.random.permutation(bg_inds.shape[0])).type_as(gt_boxes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.shape[0]-num_bg]]
                labels[i][disable_inds] = -1

        # transform each inside anchor to most related gt_boxes
        offset = torch.arange(batch_size) * gt_boxes.shape[1]
        argmax_overlaps = argmax_overlaps + offset[:, None].type_as(argmax_overlaps)
        # compute each inside anchor deltas to most related gt_boxes
        bbox_delta_targets = bbox_transform(
            anchors,
            gt_boxes.view(-1, 5)[argmax_overlaps.view(-1)].view(batch_size, -1, 5)[..., :4]
        )

        # set weight, positive inside weight: 1, postive outside weight: 1 / 256
        bbox_inside_weights[labels == 1] = self.rpn_bbox_inside_weight

        num_examples = torch.sum(labels[batch_size - 1] >= 0)
        positive_weights = 1.0 / num_examples.item()
        negative_weights = 1.0 / num_examples.item()
        bbox_outside_weights[labels == 1] = positive_weights
        bbox_outside_weights[labels == 0] = negative_weights

        total_anchors = int(K * A)
        labels = unmap(labels, total_anchors, inside_ids, batch_size, fill=-1)
        bbox_delta_targets = unmap(bbox_delta_targets, total_anchors, inside_ids, batch_size, fill=0)
        bbox_inside_weights = unmap(bbox_inside_weights, total_anchors, inside_ids, batch_size, fill=0)
        bbox_outside_weights = unmap(bbox_outside_weights, total_anchors, inside_ids, batch_size, fill=0)

        # return labels, bbox_targets, bbox_inside_weights, bbox_outside_weights
        height, width = scores.shape[2:4]
        labels = labels.view(batch_size, height, width, A).permute(0, 3, 1, 2).contiguous()
        bbox_delta_targets = bbox_delta_targets.view(batch_size, height, width, A * 4).permute(0, 3, 1, 2).contiguous()

        bbox_inside_weights = bbox_inside_weights[..., None].expand(batch_size, total_anchors, 4).contiguous()
        bbox_inside_weights = bbox_inside_weights.view(batch_size, height, width, A * 4).permute(0, 3, 1, 2).contiguous()

        bbox_outside_weights = bbox_outside_weights[..., None].expand(batch_size, total_anchors, 4).contiguous()
        bbox_outside_weights = bbox_outside_weights.view(batch_size, height, width, A * 4).permute(0, 3, 1, 2).contiguous()

        # only operation, no gradient
        return {
            "rpn_label": labels.detach(),
            "rpn_bbox_delta_target": bbox_delta_targets.detach(),
            "rpn_bbox_inside_weight": bbox_inside_weights.detach(),
            "rpn_bbox_outside_weight": bbox_outside_weights.detach(),
        }


class ProposalTargetLayer(nn.Module):
    """ proposal target layer """
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        self.bbox_means = torch.as_tensor([0.0, 0.0, 0.0, 0.0])
        self.bbox_stds = torch.as_tensor([0.1, 0.1, 0.2, 0.2])
        self.bbox_weights = torch.as_tensor([1.0, 1.0, 1.0, 1.0])

        self.fg_thresh = 0.5
        self.bg_thresh_lo = 0.1
        self.bg_thresh_hi = 0.5

        self.roi_batch_size = 128
        self.fg_frac = 0.25

    def forward(
        self,
        rois: Float[Tensor, "B M 5"], # [batch_idx, bbox]
        gt_boxes: Float[Tensor, "B N 5"], # [bbox, cls]
    ) -> Tuple[
        Float[Tensor, "B 128 5"],
        Float[Tensor, "B 128"],
        Float[Tensor, "B 128 4"],
        Float[Tensor, "B 128 4"],
        Float[Tensor, "B 128 4"]
    ]:
        gt_boxes_append = gt_boxes.new_zeros(gt_boxes.shape)
        gt_boxes_append[..., 1:5] = gt_boxes[..., :4]

        # include ground-truth boxes in candidate rois [B, M+N, 5]
        all_rois = torch.cat([rois, gt_boxes_append], 1)

        rois_per_image = self.roi_batch_size
        fg_rois_per_image = int(np.round(self.fg_frac * rois_per_image))

        labels, rois, bbox_delta_targets, bbox_inside_weights = self.sample_rois(
            all_rois,
            gt_boxes,
            fg_rois_per_image,
            rois_per_image
        )
        bbox_outside_weights = (bbox_inside_weights > 0).float()
        return rois, labels, bbox_delta_targets, bbox_inside_weights, bbox_outside_weights

    def compute_bbox_regression_labels(
        self,
        bbox_delta_target: Float[Tensor, "B N 4"],
        label: Float[Tensor, "B M"],
    ) -> Tuple[Float[Tensor, "B M 4"], Float[Tensor, "B N 4"]]:
        batch_size = label.shape[0]
        rois_per_image = label.shape[1]

        targets = bbox_delta_target.new_zeros(batch_size, rois_per_image, 4)
        inside_weights = bbox_delta_target.new_zeros(bbox_delta_target.shape)

        for i in range(batch_size):
            if label[i].sum() == 0:
                continue
            inds = torch.nonzero(label[i] > 0).view(-1)
            targets[i, inds] = bbox_delta_target[i, inds]
            inside_weights[i, inds] = self.bbox_weights

        return targets, inside_weights

    def compute_targets(
        self,
        rois: Float[Tensor, "B K 4"],
        gt: Float[Tensor, "B N 4"],
    ) -> Float[Tensor, "B N 4"]:
        targets = bbox_transform(rois, gt)
        targets = (targets - self.bbox_means[None, None]) / self.bbox_stds[None, None]
        return targets

    def sample_rois(
        self,
        rois: Float[Tensor, "B K 5"], # [batch_idx, bbox]
        gt_boxes: Float[Tensor, "B N 5"], # [bbox, cls]
        fg_rois_per_image: int,
        rois_per_image: int,
    ) -> Tuple[
        Float[Tensor, "B 128"],
        Float[Tensor, "B 128 5"],
        Float[Tensor, "B 128 4"],
        Float[Tensor, "B 128 4"]
    ]:
        batch_size = overlaps.shape[0]

        overlaps = bbox_overlaps(rois[..., 1:5], gt_boxes[..., :4])
        max_overlaps, gt_assignment = torch.max(overlaps, 2)

        offset = torch.arange(batch_size) * gt_boxes.shape[1]
        offset = offset[None].type_as(gt_assignment) + gt_assignment
        labels = gt_boxes[..., 4].contiguous().view(-1)[offset.view(-1)].view(batch_size, -1)
        
        labels_batch = labels.new_zeros(batch_size, rois_per_image)
        rois_batch  = rois.new_zeros(batch_size, rois_per_image, 5)
        gt_rois_batch = rois.new_zeros(batch_size, rois_per_image, 5)
        # Guard against the case when an image has fewer than max_fg_rois_per_image
        # foreground RoIs
        for i in range(batch_size):

            fg_inds = torch.nonzero(max_overlaps[i] >= self.fg_thresh).view(-1)
            fg_num_rois = fg_inds.numel()

            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = torch.nonzero((max_overlaps[i] < self.bg_thresh_hi) &
                                    (max_overlaps[i] >= self.bg_thresh_lo)).view(-1)
            bg_num_rois = bg_inds.numel()

            if fg_num_rois > 0 and bg_num_rois > 0:
                # sampling fg
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

                # sampling bg
                bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
                rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                bg_inds = bg_inds[rand_num]

            elif fg_num_rois > 0 and bg_num_rois == 0:
                # sampling fg
                rand_num = np.floor(np.random.rand(rois_per_image) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = rois_per_image
                bg_rois_per_this_image = 0

            elif bg_num_rois > 0 and fg_num_rois == 0:
                # sampling bg
                rand_num = np.floor(np.random.rand(rois_per_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()

                bg_inds = bg_inds[rand_num]
                bg_rois_per_this_image = rois_per_image
                fg_rois_per_this_image = 0

            else:
                raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")

            # The indices that we're selecting (both fg and bg)
            keep_inds = torch.cat([fg_inds, bg_inds], 0)

            # Select sampled values from various arrays:
            labels_batch[i].copy_(labels[i][keep_inds])

            # Clamp labels for the background RoIs to 0
            if fg_rois_per_this_image < rois_per_image:
                labels_batch[i][fg_rois_per_this_image:] = 0

            rois_batch[i] = rois[i][keep_inds]
            rois_batch[i, :, 0] = i

            gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]]

        bbox_delta_targets = self.compute_targets(rois_batch[..., 1:5], gt_rois_batch[..., :4])
        bbox_delta_targets, bbox_inside_weights = \
            self.compute_bbox_regression_labels(bbox_delta_targets, labels_batch)

        return labels_batch, rois_batch, bbox_delta_targets, bbox_inside_weights
