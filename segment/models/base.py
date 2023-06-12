import os
import json
import glob

from segment.data.base import BaseDataModule
from segment.utils.typing import *


class BaseSystem:
    def __init__(
        self,
        datamodule: BaseDataModule,
        resume: bool
    ) -> None:
        pass

    def save(self, name: str) -> None:
        raise NotImplementedError

    def restore(self, name: str) -> None:
        raise NotImplementedError

    def fit(self) -> None:
        raise NotImplementedError
    
    def inference(self) -> None:
        raise NotImplementedError
    
    def evaluate(self) -> None:
        raise NotImplementedError

    def visualize(self) -> None:
        raise NotImplementedError
    
    def fit_pose(self) -> None:
        raise NotImplementedError
