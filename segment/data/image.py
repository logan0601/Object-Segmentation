import os
import json
import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm

from segment import register
from segment.data.base import BaseDataModule
from segment.utils.config import cfg
from segment.utils.typing import *


class ImageDataset(Dataset):
    """ image dataset for 2D detection """
    def __init__(
        self,
        split: str,
    ) -> None:
        super().__init__()
        self.split = split
        # meta
        if self.split in ["train", "val"]:
            with open(os.path.join(cfg.split_dir, f"{split}.txt")) as f:
                prefixs = sorted([line.strip() for line in f if line.strip()])
                self.meta = [
                    os.path.join(
                        os.path.dirname(cfg.train_dir),
                        "process",
                        prefix + "_meta.json"
                    ) 
                    for prefix in prefixs
                ]
        elif self.split in ["test"]:
            self.meta = sorted(glob.glob(
                os.path.join(os.path.dirname(cfg.test_dir), "process", "*_meta.json")))

    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        with open(self.meta[index]) as f:
            meta: dict = json.load(f)
        if "annotations" in meta:
            for obj in meta["annotations"]:
                ymin, xmin, ymax, xmax = obj["bbox"]
                obj["bbox"] = [xmin, ymin, xmax, ymax]
        return meta
    
    def __call__(self) -> List[Dict[str, Any]]:
        ret_dict = []
        for met in tqdm(self.meta):
            met = json.load(open(met, "r"))
            if "annotations" in met:
                for obj in met["annotations"]:
                    ymin, xmin, ymax, xmax = obj["bbox"]
                    obj["bbox"] = [xmin, ymin, xmax, ymax]
            ret_dict.append(met)
        return ret_dict


@register("image-datamodule")
class ImageDataModule(BaseDataModule):
    def __init__(
        self,
        stage: Optional[str] = None
    ) -> None:
        self.stage = stage
        self.batch_size = cfg.batch_size
        if self.stage in [None, "fit"]:
            self.train_dataset = ImageDataset(split="train")
        if self.stage in [None, "fit", "validate"]:
            self.val_dataset = ImageDataset(split="val")
        if self.stage in [None, "test"]:
            self.test_dataset = ImageDataset(split="test")
        
    def general_loader(self, dataset: Dataset, batch_size: int) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=cfg.num_workers,
            batch_size=batch_size
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=self.batch_size
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1
        )

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1
        )
