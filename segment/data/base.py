import os
import json

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from segment.utils.typing import *


class BaseDataModule:
    """ define base datamodule """
    batch_size: int
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    apply_dataset: Dataset

    def general_loader(
        self,
        dataset: Dataset,
        batch_size: int
    ) -> DataLoader:
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError
    
    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError
    
    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def apply_dataloader(self) -> DataLoader:
        raise NotImplementedError
