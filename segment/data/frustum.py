import os
import json
import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

import segment
from segment import register
from segment.data.base import BaseDataModule
from segment.utils.config import cfg
from segment.utils.typing import *


class FrustumDataset(Dataset):
    """ point dataset for 3D detection """
    def __init__(
        self,
        split: str,
    ) -> None:
        super().__init__()
        self.split = split
        
        if self.split in ["train"]:
            self.max_size = 1000
        elif self.split in ["val"]:
            self.max_size = 2500
        elif self.split in ["test"]:
            self.max_size = 1000

        # meta
        if self.split in ["train", "val"]:
            with open(os.path.join(cfg.split_dir, f"{split}.txt")) as f:
                prefixs = sorted([line.strip() for line in f if line.strip()])
                self.meta = [
                    os.path.join(
                        os.path.dirname(cfg.train_dir),
                        "process",
                        prefix + "_data.pth"
                    )
                    for prefix in prefixs
                ]
        elif self.split in ["test"]:
            self.meta = sorted(glob.glob(
                os.path.join(os.path.dirname(cfg.test_dir), "process", "*_data.pth")))
            self.proposals = json.load(open(os.path.join(
                cfg.output_dir, "proposal_test.json"
            )))

        segment.info("Loading frustum points...")
        self.load_idx = 0
        self.load_tot = len(self.meta) // self.max_size
        if self.split in ["train"]:
            self.rand_list = np.random.permutation(len(self.meta))
        elif self.split in ["val", "test"]:
            self.rand_list = np.arange(len(self.meta))
        self.tem_list = [torch.load(self.meta[ind]) for ind in tqdm(self.rand_list[:self.max_size])]

    def __len__(self) -> int:
        return self.max_size
    
    def get_prefix(self, index) -> str:
        return os.path.basename(
            self.meta[self.rand_list[self.load_idx * self.max_size + index]]
        )[:-9]

    def step(self) -> None:
        segment.info("Loading frustum points...")
        self.load_idx = (self.load_idx + 1) % self.load_tot
        self.tem_list = [torch.load(self.meta[ind]) for ind in tqdm(
            self.rand_list[(self.load_idx*self.max_size):((self.load_idx+1)*self.max_size)]
        )]

    def __getitem__(self, index) -> Dict[str, Any]:
        if self.split in ["train", "val"]:
            rgb, dep, lab, intrin, bbox = self.tem_list[index]
            rgb = rgb.float() / 255
            dep = dep.float() / 1000
            dif_y = bbox[:, 2] - bbox[:, 0]
            dif_x = bbox[:, 3] - bbox[:, 1]
            lbl = bbox[:, 4]
            mask = (dif_y > 0) * (dif_x > 0) * (lbl < cfg.num_objects)
            bbox = bbox[mask]
            return {
                "id": index,
                "rgb": rgb,
                "dep": dep,
                "intrin": intrin,
                "bbox": bbox,
                "label": lab.clamp_(0, cfg.num_objects)
            }
        elif self.split in ["test"]:
            rgb, dep, intrin = self.tem_list[index]
            rgb = rgb.float() / 255
            dep = dep.float() / 1000
            return {
                "id": index,
                "rgb": rgb,
                "dep": dep,
                "intrin": intrin
            }


@register("frustum-datamodule")
class FrustumDataModule(BaseDataModule):
    def __init__(
        self,
        stage: Optional[str] = None
    ) -> None:
        self.stage = stage
        if self.stage in [None, "fit"]:
            self.train_dataset = FrustumDataset(split="train")
        if self.stage in [None, "fit", "validate"]:
            self.val_dataset = FrustumDataset(split="val")
        if self.stage in [None, "test"]:
            self.test_dataset = FrustumDataset(split="test")
        
    def general_loader(self, dataset: Dataset, batch_size: int) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=cfg.num_workers,
            batch_size=batch_size,
            shuffle=False
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=1
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1
        )

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1
        )
