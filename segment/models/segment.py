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
from segment.utils.ops import (
    backproject,
    normalize,
    scale_tensor,
    compute_intersection_over_union,
    generate_bbox,
    euler_to_rotation,
    eval_pose
)
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
            score = scale_tensor(score, (score.min(), score.max()), (-1, 1))
            segs.append(score)
        return segs


class FrustumSegmentationNet(nn.Module):
    def __init__(self, inp_dim=3, hid_dim=128, oup_dim=512) -> None:
        super(FrustumSegmentationNet, self).__init__()
        self.f1 = nn.Linear(3 + inp_dim, 3 + hid_dim)
        self.f2 = nn.Linear(3 + hid_dim, 3 + hid_dim)
        self.f3 = nn.Linear(3 + hid_dim, 3 + hid_dim)
        self.h1 = nn.Linear(3 + hid_dim, 3 + oup_dim)
        self.h2 = nn.Linear(3 + oup_dim, 3 + oup_dim)
        self.h3 = nn.Linear(3 + oup_dim, 3 + oup_dim)
        
        self.get_score = nn.Linear(9 + inp_dim + hid_dim + oup_dim, 1)
    
    def pointnet2(self, inp):
        hidden = F.relu(self.f1(inp))
        hidden = F.relu(self.f2(hidden))
        hidden = F.relu(self.f3(hidden))
        x = torch.max(hidden, dim=0)[0]
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        return hidden, x
    
    def center(self, pc, type='xyz'):
        if type == 'xyz':
            xyz = pc[..., :3]
            feats = pc[..., 3:]
            xyz = xyz - torch.mean(xyz, dim=-2, keepdim=True)
            length = (xyz.max(dim=-2, keepdim=True)[0] - xyz.min(dim=-2, keepdim=True)[0]).clamp(min=1e-5)
            
            xyz = xyz / length # scale
            centered_pc = torch.cat([xyz, feats], dim=-1)
            return centered_pc
        else:
            pc = pc - torch.mean(pc, dim=-2, keepdim=True)
            length = (pc.max(dim=-2, keepdim=True)[0] - pc.min(dim=-2, keepdim=True)[0]).clamp(min=1e-5)
            pc = pc / length
            return pc

    def image2pc(self, depth, intrinsic):
        """
        Takes in the cropped depth and intrinsic data, return the pointcloud. 
        """
        z = depth
        v, u = np.indices(z.shape)
        v = torch.from_numpy(v).cuda()
        u = torch.from_numpy(u).cuda()
        uv1 = torch.stack([u + 0.5, v + 0.5, torch.ones_like(z)], axis=-1)
        coords = uv1 @ torch.linalg.inv(intrinsic).T * z[..., None]  # [H, W, 3]
        return coords

    def forward(self, rgb, depth, bbox, intrinsic):
        segs = []
        for (x1, y1, x2, y2, _) in bbox:
            # if x1 >= x2-1 or y1 >= y2-1: continue
            # Cropping and lifting
            cropped_pc = self.image2pc(depth[x1 : x2, y1 : y2], intrinsic)
            # 3D PointCloud Segmentation
            x = torch.cat([cropped_pc, rgb[x1 : x2, y1 : y2]], dim=-1)
            x = self.center(x)
            orig_shape = x.shape
            x = x.view((-1, orig_shape[-1]))

            h_feats, abs_feats = self.pointnet2(x)
            seg = torch.cat([x, h_feats, torch.repeat_interleave(abs_feats.unsqueeze(0), x.size(0), dim=0)], dim=-1)

            assert x.size(0) == seg.size(0) > 0
            seg = seg.view((*orig_shape[:-1], -1)) # H, W, 9 + i + h + o

            seg = self.get_score(seg).squeeze(-1) # > 0 yes; < 0 no
            A, B = seg.max(), seg.min()
            seg = seg - (A + B) / 2
            if A > B:
                seg = 2 * seg / (A - B)
            segs.append(seg)
        return segs


class STNNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_dim: int = 6
        self.hidden_dim: int = cfg.hidden_dim
        self.output_dim: int = cfg.output_dim
        self.embed = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.segment = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
        )
        self.score = nn.Linear(self.output_dim, 9)

    def forward(
        self,
        rgb: Float[Tensor, "H W 3"],
        dep: Float[Tensor, "H W"],
        bbox: Float[Tensor, "M 5"],
        intrin: Float[Tensor, "3 3"],
        label: Int[Tensor, "H W"],
    ) -> List[Float[Tensor, "9"]]:
        out = []
        for box in bbox:
            y_lo, x_lo, y_hi, x_hi, ids = box

            # only consider object points
            mask = (label[y_lo:y_hi, x_lo:x_hi] == ids).view(-1)
            pnts = backproject(dep[y_lo:y_hi, x_lo:x_hi], intrin).view(-1, 3)[mask]
            feat = rgb[y_lo:y_hi, x_lo:x_hi].contiguous().view(-1, 3)[mask]

            # check
            center = pnts.mean(dim=0)
            
            pnts = normalize(pnts, dim=0)
            feat = torch.cat([pnts, feat], dim=-1)
            feat = self.embed(feat)
            feat = torch.max(feat, dim=0)[0]
            feat = self.segment(feat)
            feat = self.score(feat) # 9
            
            box_3d = generate_bbox(pnts).cuda()
            size = feat[:3][None]
            cres = feat[3:6][None]
            angle = feat[6:]
            rotation = euler_to_rotation(angle).cuda()

            corner = (box_3d * size) @ rotation.T + center + cres
            pose = torch.zeros((4, 4), dtype=torch.float32).cuda()
            pose[:3, :3] = rotation
            pose[:3, 3] = center + cres
            pose[3, 3] = 1.0
            out.append((corner, pose))

        return out


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
        self.best_iou = 0
        self.metrics = None
        self.output_dir = os.path.join(cfg.output_dir, "frustum-pointnet-v2")
        os.makedirs(self.output_dir, exist_ok=True)

        self.segmentor = FrustumSegmentationNet().cuda()
        self.estimator = STNNet().cuda()
        self.optimizer = optim.Adam(self.segmentor.parameters(), lr=1e-3)
        self.writer = SummaryWriter(os.path.join(
            self.output_dir, "log-{}".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
        ))

    def restore(
        self,
        name: str
    ) -> None:
        segment.info(f"Restoring from checkpoint {name}")
        model_dict: dict = torch.load(os.path.join(self.output_dir, name))
        self.current_step = model_dict["step"]
        self.current_epoch = model_dict["epoch"]
        self.best_iou = model_dict.get("iou", 0)
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
            "iou": self.best_iou,
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
            for batch in tqdm(dataloader):
                rgb = batch["rgb"][0].cuda()
                dep = batch["dep"][0].cuda()
                bbox = batch["bbox"][0].cuda()
                intrin = batch["intrin"][0].cuda()
                label = batch["label"][0].cuda().long()

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

            self.current_epoch += 1
            
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
                if val_iou > self.best_iou:
                    self.inference()
                    segment.info("Updating best model...")
                    self.best_iou = val_iou
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
        os.makedirs(os.path.join(self.output_dir, "res"), exist_ok=True)
        dataset: FrustumDataset = self.datamodule.test_dataset
        dataloader = self.datamodule.test_dataloader()
        for batch in tqdm(dataloader):
            torch.cuda.empty_cache()
            rgb = batch["rgb"][0].cuda()
            dep = batch["dep"][0].cuda()
            bbox = batch["bbox"][0].cuda()
            intrin = batch["intrin"][0].cuda()
            score = batch["score"][0].cuda()

            segs = self.segmentor(rgb, dep, bbox, intrin)
            pred = torch.zeros_like(dep, dtype=torch.int32)
            conf = torch.zeros_like(dep, dtype=torch.float32)
            mask = torch.ones_like(dep, dtype=torch.bool)
            for box, sco, seg in zip(bbox, score, segs):
                y_lo, x_lo, y_hi, x_hi, ids = box
                seg = seg.detach()
                pred[y_lo:y_hi, x_lo:x_hi] = ids * (sco * seg > conf[y_lo:y_hi, x_lo:x_hi])
                conf[y_lo:y_hi, x_lo:x_hi] = torch.maximum(sco * seg, conf[y_lo:y_hi, x_lo:x_hi])
                mask[y_lo:y_hi, x_lo:x_hi] *= (seg <= 0)

            pred[mask] = cfg.num_objects
            pred = pred.detach().cpu().numpy()
            prefix = dataset.get_prefix(batch["id"][0])
            Image.fromarray(pred).convert('P').save(
                os.path.join(self.output_dir, "res", prefix + "_label_kinect.png")
            )
            pred = np.expand_dims(pred, axis=2).repeat(3, axis=2).astype(np.uint8)
            Image.fromarray(
                np.concatenate([(rgb.detach().cpu().numpy() * 255).astype(np.uint8), pred], axis=1)
            ).save(
                os.path.join(self.output_dir, "res", prefix + "_concat.png")
            )

    def fit_pose(
        self,
    ) -> None:
        segment.info("Fitting 3d poses...")
        # fix segmentor
        for p in self.segmentor.parameters():
            p.requires_grad = False

        optimizer = optim.Adam(self.estimator.parameters(), lr=1e-3)
        dataloader = self.datamodule.apply_dataloader()
        for epoch in range(cfg.max_epochs):
            avg_loss = 0
            avg_r_diff = 0
            avg_t_diff = 0
            for batch in tqdm(dataloader):
                torch.cuda.empty_cache()
                rgb = batch["rgb"][0].cuda()
                dep = batch["dep"][0].cuda()
                bbox = batch["bbox"][0].cuda()
                intrin = batch["intrin"][0].cuda()
                label = batch["label"][0].cuda().long()
                corner = batch["corner"][0].cuda()
                pose = batch["pose"][0].cuda()

                out = self.estimator(rgb, dep, bbox, intrin, label)
                loss = 0
                r_diff, t_diff = 0, 0
                for (box, pred), cor, pos in zip(out, corner, pose):
                    loss += F.mse_loss(box, cor)
                    r_, t_ = eval_pose(pred.detach().cpu().numpy(), pos.detach().cpu().numpy())
                    r_diff += r_
                    t_diff += t_
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                avg_r_diff += r_diff.item() / len(out)
                avg_t_diff += t_diff.item() / len(out)

            avg_loss /= len(dataloader)
            avg_r_diff /= len(dataloader)
            avg_t_diff /= len(dataloader)

            segment.info(
                f"Epoch: {epoch}/{cfg.max_epochs}, train loss: {avg_loss:.4f}, t_diff: {avg_t_diff:.4f}, r_diff: {avg_r_diff:.4f}")
