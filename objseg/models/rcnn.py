import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import roi_pool

from objseg.utils.ops import ProposalLayer, AnchorTargetLayer, ProposalTargetLayer
from objseg.models.loss import smooth_l1_loss
from objseg.utils.typing import *
    

class RPN(nn.Module):
    """ region proposal network """
    def __init__(self) -> None:
        super().__init__()

        # hardcoded scales and ratios
        self.anchor_scales = torch.as_tensor([8, 16, 32]).float()
        self.anchor_ratios = torch.as_tensor([0.5, 1, 2]).float()
        self.feat_stride = 16

        # conv into feature map
        self.rpn_conv = nn.Conv2d(512, 512, 3, 1, 1, bias=True)

        # define bg/fg classification score
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2
        self.score_conv = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4
        self.bbox_conv = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # define proposal
        self.proposal_layer = ProposalLayer(
            self.feat_stride, self.anchor_scales, self.anchor_ratios)
        
        # define proposal loss
        self.anchor_target_layer = AnchorTargetLayer(
            self.feat_stride, self.anchor_scales, self.anchor_ratios)
    
    @staticmethod
    def reshape_layer(
        x: Float[Tensor, "B C H W"],
        d: int
    ) -> Float[Tensor, "B d h W"]:
        """ reshape dim=1 """
        shape = x.shape
        x = x.view(
            shape[0],
            int(d),
            int(float(shape[1] * shape[2]) / float(d)),
            shape[3]
        )
        return x
    
    @staticmethod
    def compute_loss(
        rpn_cls_score: Float[Tensor, "B 18 H W"],
        rpn_bbox_delta_pred: Float[Tensor, "B 36 H W"],
        rpn_label: Float[Tensor, "B 9 H W"],
        rpn_bbox_delta_target: Float[Tensor, "B 36 H W"],
        rpn_bbox_inside_weight: Float[Tensor, "B 36 H W"],
        rpn_bbox_outside_weight: Float[Tensor, "B 36 H W"],
    ) -> Tuple[Float[Tensor, "1"], Float[Tensor, "1"]]:
        batch_size = rpn_cls_score.shape[0]

        # compute classification loss
        rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
        rpn_label = rpn_label.view(batch_size, -1)

        rpn_keep = rpn_label.view(-1).ne(-1).nonzero().view(-1)
        rpn_cls_score = torch.index_select(rpn_cls_score.view(-1, 2), 0, rpn_keep)
        rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep).long()
        rpn_cls_loss = F.cross_entropy(rpn_cls_score, rpn_label)

        # compute bbox regression loss
        rpn_box_loss = smooth_l1_loss(
            rpn_bbox_delta_pred,
            rpn_bbox_delta_target,
            rpn_bbox_inside_weight,
            rpn_bbox_outside_weight,
            sigma=3.0,
            dims=[1, 2, 3]
        )
        return rpn_cls_loss, rpn_box_loss

    def forward(
        self,
        features: Float[Tensor, "B 512 H W"],
        gt_boxes: Optional[Float[Tensor, "B K 4"]],
        img_info: Float[Tensor, "2"],
    ) -> Tuple[Float[Tensor, "B M 5"], Float[Tensor, "1"], Float[Tensor, "1"]]:
        # convert pixels to feature maps
        rpn_conv = F.relu(self.rpn_conv(features), inplace=True)

        # compute cls score
        rpn_cls_score = self.score_conv(rpn_conv)
        rpn_cls_score_reshape = self.reshape_layer(rpn_cls_score, 2)
        rpn_cls_prob = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob_reshape = self.reshape_layer(rpn_cls_prob, self.nc_score_out)

        # compute rpn boxes deltas
        rpn_bbox_delta = self.bbox_conv(rpn_conv)

        # compute proposal
        state = "train" if self.training else "test"
        rois = self.proposal_layer(rpn_cls_prob_reshape, rpn_bbox_delta, img_info, state)

        # generate training labels and compute rpn loss
        rpn_cls_loss, rpn_box_loss = 0, 0
        if self.training:
            rpn_data = self.anchor_target_layer(rpn_cls_score, gt_boxes, img_info)
            rpn_cls_loss, rpn_box_loss = self.compute_loss(
                rpn_cls_score_reshape,
                rpn_bbox_delta,
                **rpn_data
            )
        
        # roi [B, M, 5] [batch_idx, bbox]
        return rois, rpn_cls_loss, rpn_box_loss
    

class RoIPool(nn.Module):
    """ RoI Pooling """
    def __init__(self, pooled_height: int, pooled_width: int):
        super().__init__()
        self.pooled_h = pooled_height
        self.pooled_w = pooled_width
    
    def forward(
        self,
        features: Float[Tensor, "B 512 H W"],
        rois: Float[Tensor, "B N 5"],
    ):
        batch_size = rois.shape[0]
        output = []
        for i in range(batch_size):
            output.append(
                roi_pool(
                    features[i],
                    rois[i],
                    output_size=(self.pooled_h, self.pooled_w)
                )
            )
        
        output = torch.stack(output, 0)
        return output


class FasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.n_classes = n_classes

        # define backbone VGG16
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(vgg.features._modules.values())[:-1])
        # fix first 10 layers
        for layer in range(10):
            for p in self.backbone[layer].parameters():
                p.requires_grad = False

        # generate rigion proposal in feature map
        self.rpn = RPN()

        # compute target proposal
        self.proposal_target_layer = ProposalTargetLayer(n_classes)

        # pool proposal to fix size
        self.roi_pool = RoIPool(7, 7)

        # define mlp
        self.mlp = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

        # define cls classifier
        self.cls_classifier = nn.Linear(4096, self.n_classes)

        # define bbox classifier
        self.bbox_predictor = nn.Linear(4096, self.n_classes * 4)
    
    def forward(
        self,
        images: Float[Tensor, "B 3 H W"],
        gt_boxes: Float[Tensor, "B M 5"],
        img_info: Float[Tensor, "2"],
    ) -> Dict[str, Any]:
        batch_size = images.shape[0]

        # use backbone to get feature map [B 512, H/16, W/16]
        base_feat = self.backbone(images)

        # use regoin proposal networks to obtain rois
        rois, rpn_cls_loss, rpn_box_loss = self.rpn(base_feat, gt_boxes, img_info)

        # if training phase, use ground-truth for refine
        rois_label: Tensor = None
        rois_delta_target: Tensor = None
        rois_inside_ws: Tensor = None
        rois_outside_ws: Tensor = None
        if self.training:
            (
                rois, # [B, 128, 5]
                rois_label, # [B, 128]
                rois_delta_target, # [B, 128, 4]
                rois_inside_ws, # [B, 128, 4]
                rois_outside_ws # [B, 128, 4]
            ) = self.proposal_target_layer(rois, gt_boxes)

            rois_label = rois_label.view(-1).long()
            rois_delta_target = rois_delta_target.view(-1, rois_delta_target.shape[2])
            rois_inside_ws = rois_inside_ws.view(-1, rois_inside_ws.shape[2])
            rois_outside_ws = rois_outside_ws.view(-1, rois_outside_ws.shape[2])

        pooled_feat = self.roi_pool(base_feat, rois.view(-1, 5))
        pooled_feat = self.mlp(pooled_feat.view(pooled_feat.shape[0], -1))

        # compute bbox offset
        bbox_pred = self.bbox_predictor(pooled_feat)
        if self.training:
            bbox_pred_reshape = bbox_pred.view(bbox_pred.shape[0], -1, 4)
            bbox_pred_select = torch.gather(
                bbox_pred_reshape,
                1,
                rois_label[..., None, None].expand(rois_label.shape[0], 1, 4)
            )
            bbox_pred = bbox_pred_select.squeeze(1)
        
        # compute cls classification prob
        cls_score = self.cls_classifier(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        rcnn_cls_loss, rcnn_box_loss = 0, 0
        if self.training:
            # cls loss
            rcnn_cls_loss = F.cross_entropy(cls_score, rois_label)

            # bbox regression L1 loss
            rcnn_box_loss = smooth_l1_loss(
                bbox_pred,
                rois_delta_target,
                rois_inside_ws,
                rois_outside_ws
            )
        
        cls_prob = cls_prob.view(batch_size, rois.shape[1], -1)
        bbox_pred = bbox_pred.view(batch_size, rois.shape[1], -1)

        return {
            "rois": rois,
            "cls_prob": cls_prob,
            "bbox_pred": bbox_pred,
            "rpn_cls_loss": rpn_cls_loss,
            "rpn_box_loss": rpn_box_loss,
            "rcnn_cls_loss": rcnn_cls_loss,
            "rcnn_box_loss": rcnn_box_loss,
            "rois_label": rois_label
        }
