import os
import numpy as np

from segment.utils.typing import *


class ExperimentConfig:
    trial_name: str = ""

    # image
    height: int = 720
    width: int = 1280

    # objects
    num_objects: int = 79
    tot_objects: int = 82

    # training
    learning_rate: float = 2.5e-4
    max_iters: int = 80000
    max_epochs: int = 160

    # data
    batch_size: int = 4
    num_workers: int = 4
    max_bbox: int = 60

    # dataset
    train_dir: str = "datasets/training_data/data"
    test_dir: str = "datasets/testing_data/data"
    split_dir: str = "datasets/training_data/splits"
    output_dir: str = "outputs/"

    # freq
    eval_every: int = 2000
    eval_every_3d: int = 10

    # pointnet
    hidden_dim: int = 128 + 3
    output_dim: int = 512 + 3


cfg: ExperimentConfig = ExperimentConfig()
