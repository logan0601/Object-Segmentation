import os
from datetime import datetime

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from tensorboardX import SummaryWriter

import segment
from segment import register
from segment.data.frustum import FrustumDataset, FrustumDataModule
from segment.models.base import BaseSystem
from segment.utils.config import cfg
from segment.utils.ops import backproject, normalize, compute_intersection_over_union
from segment.utils.typing import *


class PointNet(nn.Module):
    """ 3d segmentation network """
    def __init__(self) -> None:
        super().__init__()
        # FIXME: hardcoded input_dim
        self.input_dim: int = 6
        self.hidden_dim: int = cfg.hidden_dim
        self.output_dim: int = cfg.output_dim
        self.embed = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        self.segment = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(self.input_dim + self.hidden_dim + self.output_dim, 1)

    def forward(
        self,
        rgb: Float[Tensor, "H W 3"],
        dep: Float[Tensor, "H W"],
        bbox: Float[Tensor, "M 5"],
        intrin: Float[Tensor, "3 3"]
    ) -> List[Float[Tensor, "H W"]]:
        # FIXME: only apply for batch_size=1 due to frustumdataloader
        num_bbox = bbox.shape[0]
        segs = []
        for i in range(num_bbox):
            y_lo, x_lo, y_hi, x_hi = bbox[i, :4]
            pnts = backproject(dep[y_lo:y_hi, x_lo:x_hi], intrin)
            pnts = normalize(pnts)
            feat = torch.cat([pnts, rgb[y_lo:y_hi, x_lo:x_hi]], dim=-1).view(-1, 6)
            hidden_feat = self.embed(feat)
            pooled_feat = torch.max(hidden_feat, dim=0)[0]
            output_feat = self.segment(pooled_feat)
            concat_feat = torch.cat([
                feat, hidden_feat, output_feat[None].repeat(feat.shape[0], 1)
            ], dim=-1)
            score = self.classifier(concat_feat)[:, 0]
            score = score.view(pnts.shape[:2])
            score_max = score.max()
            score_min = score.min()
            score = score - (score_max + score_min) / 2.0
            if score_max > score_min:
                score = score * 2.0 / (score_max - score_min)
            segs.append(score)
        return segs


@register("3d-segmentor")
class Segmentor(BaseSystem):
    """ 3D segmentor """
    def __init__(
        self,
        datamodule: FrustumDataModule,
        resume: bool
    ) -> None:
        self.datamodule = datamodule
        self.resume = resume
        self.current_epoch = 0
        self.current_step = 0
        self.metrics = None
        self.output_dir = os.path.join(cfg.output_dir, "frustum-pointnet")
        os.makedirs(self.output_dir, exist_ok=True)

        self.segmentor = PointNet().cuda()
        self.optimizer = optim.Adam(self.segmentor.parameters(), lr=1e-3)
        self.writer = SummaryWriter(os.path.join(
            self.output_dir, "log-{}".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
        ))

    def restore(
        self,
        name: str
    ) -> None:
        segment.info(f"Restoring from checkpoint {name}")
        model_dict = torch.load(os.path.join(self.output_dir, name))
        self.current_step = model_dict["step"]
        self.current_epoch = model_dict["epoch"]
        self.segmentor.load_state_dict(model_dict["model"])
        self.optimizer.load_state_dict(model_dict["optim"])

    def save(
        self,
        name: str
    ) -> None:
        segment.info(f"Saving to checkpoint {name}")
        torch.save({
            "step": self.current_step,
            "epoch": self.current_epoch,
            "model": self.segmentor.state_dict(),
            "optim": self.optimizer.state_dict()
        }, os.path.join(self.output_dir, name))

    def fit(
        self
    ) -> None:
        segment.info("3D detector fitting...")        
        dataloader = self.datamodule.train_dataloader()
        # FIXME: only work for batch_size=1
        for epoch in range(self.current_epoch, cfg.max_epochs):

            self.segmentor.train()
            avg_loss = 0
            best_iou = 0
            for batch in tqdm(dataloader):
                rgb = batch["rgb"][0].cuda()
                dep = batch["dep"][0].cuda()
                bbox = batch["bbox"][0].cuda()
                intrin = batch["intrin"][0].cuda()
                label = batch["label"][0].cuda().long()
                label.clamp_(max=cfg.num_objects)

                segs = self.segmentor(rgb, dep, bbox, intrin)
                loss = 0
                for box, seg in zip(bbox, segs):
                    y_lo, x_lo, y_hi, x_hi, ids = box
                    gt = label[y_lo:y_hi, x_lo:x_hi]
                    loss += (torch.sum(1 - seg[gt == ids]) + torch.sum(1 + seg[gt != ids])) / ((y_hi - y_lo) * (x_hi - x_lo))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item()

                self.current_step += 1
                self.writer.add_scalar("train/loss", loss, self.current_step)
            
            avg_loss /= len(dataloader)
            segment.info(f"Epoch: {epoch}/{cfg.max_epochs}, train loss: {avg_loss:.4f}")
            self.datamodule.train_dataset.step()

            if (epoch + 1) % cfg.eval_every_3d == 0:
                self.save(f"model_{epoch:05d}.pth")
                if os.path.exists(os.path.join(self.output_dir, f"model_{epoch-cfg.eval_every_3d:05d}.pth")):
                    os.remove(
                        os.path.join(self.output_dir, f"model_{epoch-cfg.eval_every_3d:05d}.pth")
                    )

                self.evaluate()
                val_loss, val_acc, val_iou = self.metrics
                segment.info(
                    f"Epoch: {epoch}/{cfg.max_epochs}, val loss: {val_loss:.4f}, mAcc: {val_acc:.4f}, mIoU: {val_iou:.4f}"
                )
                if val_iou > best_iou:
                    segment.info("Updating best model...")
                    best_iou = val_iou
                    self.save("model_best.pth")

    def evaluate(
        self
    ) -> None:
        segment.info("3d detector evaluating...")
        self.segmentor.eval()
        dataloader = self.datamodule.val_dataloader()
        avg_loss = 0
        intersection, union, samples = [np.zeros(cfg.num_objects) for _ in range(3)]
        for batch in tqdm(dataloader):
            torch.cuda.empty_cache()
            rgb = batch["rgb"][0].cuda()
            dep = batch["dep"][0].cuda()
            bbox = batch["bbox"][0].cuda()
            intrin = batch["intrin"][0].cuda()
            label = batch["label"][0].cuda().long()
            label.clamp_(max=cfg.num_objects)

            segs = self.segmentor(rgb, dep, bbox, intrin)
            loss = 0
            pred = torch.zeros_like(dep, dtype=torch.int32)
            conf = torch.zeros_like(dep, dtype=torch.float32)
            mask = torch.zeros_like(dep, dtype=torch.bool)
            for box, seg in zip(bbox, segs):
                y_lo, x_lo, y_hi, x_hi, ids = box
                gt = label[y_lo:y_hi, x_lo:x_hi]
                seg = seg.detach()
                loss += (torch.sum(1 - seg[gt == ids]) + torch.sum(1 + seg[gt != ids])) / ((y_hi - y_lo) * (x_hi - x_lo))

                pred[y_lo:y_hi, x_lo:x_hi] = ids * (seg > conf[y_lo:y_hi, x_lo:x_hi])
                conf[y_lo:y_hi, x_lo:x_hi] = torch.maximum(seg, conf[y_lo:y_hi, x_lo:x_hi])
                mask[y_lo:y_hi, x_lo:x_hi] *= (seg <= 0)

            avg_loss += loss.item()
            pred[mask] = cfg.num_objects

            pred = pred.detach().cpu().numpy()
            label = label.cpu().numpy()
            i, u, n = compute_intersection_over_union(pred, label)
            intersection += i
            union += u
            samples += n
        
        avg_loss /= len(dataloader)
        acc = (intersection / (samples + 1e-5)) * 100.0
        mAcc = np.mean(acc[samples > 0])
        iou = (intersection / (union + 1e-5)) * 100.0
        mIoU = np.mean(iou[union > 0])
        self.metrics = (avg_loss, mAcc.item(), mIoU.item())
        self.writer.add_scalar("val/loss", avg_loss, self.current_step)
        self.writer.add_scalar("val/acc", mAcc, self.current_step)
        self.writer.add_scalar("val/iou", mIoU, self.current_step)

    def inference(
        self
    ) -> None:
        segment.info("3d detector inferencing...")
        self.segmentor.eval()
        # FIXME: hardcode output dir
        os.makedirs(os.path.join(self.output_dir, "res"))
        dataset: FrustumDataset = self.datamodule.test_dataset
        dataloader = self.datamodule.test_dataloader()
        for batch in tqdm(dataloader):
            torch.cuda.empty_cache()
            rgb = batch["rgb"][0].cuda()
            dep = batch["dep"][0].cuda()
            bbox = batch["bbox"][0].cuda()
            intrin = batch["intrin"][0].cuda()

            segs = self.segmentor(rgb, dep, bbox, intrin)
            pred = torch.zeros_like(dep, dtype=torch.int32)
            conf = torch.zeros_like(dep, dtype=torch.float32)
            mask = torch.zeros_like(dep, dtype=torch.bool)
            for box, seg in zip(bbox, segs):
                y_lo, x_lo, y_hi, x_hi, ids = box
                seg = seg.detach()
                pred[y_lo:y_hi, x_lo:x_hi] = ids * (seg > conf[y_lo:y_hi, x_lo:x_hi])
                conf[y_lo:y_hi, x_lo:x_hi] = torch.maximum(seg, conf[y_lo:y_hi, x_lo:x_hi])
                mask[y_lo:y_hi, x_lo:x_hi] *= (seg <= 0)

            pred[mask] = cfg.num_objects
            prefix = dataset.get_prefix(batch["id"][0])
            Image.fromarray(pred.detach().cpu().numpy()).save(
                os.path.join(self.output_dir, "res", prefix + "_label_kinect.png")
            )
            Image.fromarray(
                np.concatenate([(rgb.detach().cpu().numpy() * 255).astype(np.int32), pred], axis=1)
            ).save(
                os.path.join(self.output_dir, "res", prefix + "_concat.png")
            )

    def visualize(
        self
    ) -> None:
        raise NotImplementedError
