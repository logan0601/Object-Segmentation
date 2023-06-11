import os
import json
from tqdm import tqdm

from PIL import Image
import numpy as np
from torchvision.ops import batched_nms

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor

import segment
from segment import register
from segment.data.image import ImageDataModule
from segment.models.base import BaseSystem
from segment.utils.config import cfg
from segment.utils.typing import *
from segment.utils.visualize import demo


@register("2d-detector")
class Detector(BaseSystem):
    """ 2D Detector """
    def __init__(
        self,
        datamodule: ImageDataModule,
        resume: bool
    ) -> None:
        config = get_cfg()
        config.merge_from_file(
            model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
        config.DATASETS.TRAIN = ("train", )
        config.DATASETS.TEST = ("val", )
        config.EVAL_PERIOD = cfg.eval_every
        config.DATALOADER.NUM_WORKERS = cfg.num_workers
        config.SOLVER.IMS_PER_BATCH = cfg.batch_size
        config.SOLVER.BASE_LR = cfg.learning_rate
        config.SOLVER.MAX_ITER = cfg.max_iters
        config.SOLVER.STEPS = []
        config.SOLVER.CHECKPOINT_PERIOD = cfg.eval_every
        config.MODEL.ROI_HEADS.NUM_CLASSES = cfg.num_objects
        config.OUTPUT_DIR = os.path.join(cfg.output_dir, "retina_net")
        self.config = config
        self.resume = resume
        self.datamodule = datamodule

        DatasetCatalog.register("train", datamodule.train_dataset)
        DatasetCatalog.register("val", datamodule.val_dataset)
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    def save(
        self,
        name: str
    ) -> None:
        segment.info(f"Saving to checkpoint {name}")
        pass

    def restore(
        self,
        name: str
    ) -> None:
        self.config.MODEL.WEIGHTS = os.path.join(self.config.OUTPUT_DIR, name)

    def fit(
        self
    ) -> None:
        segment.info("2D detector fitting...")
        trainer = DefaultTrainer(self.config)
        trainer.resume_or_load(resume=self.resume)
        trainer.train()

    def evaluate(
        self
    ) -> None:
        segment.info("2D detector evaluating...")
        self.config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0
        predictor = DefaultPredictor(self.config)

        preds = []
        dataset = self.datamodule.val_dataset()
        for dat in tqdm(dataset):
            im = np.array(Image.open(dat["file_name"]))
            out = predictor(im)
            boxes, scores, labels = out["instances"].pred_boxes, out["instances"].scores, out["instances"].pred_classes
            boxes = np.array(boxes.tensor.cpu()).tolist()
            scores = np.array(scores.cpu()).tolist()
            labels = np.array(labels.cpu()).tolist()
            preds.append([boxes, labels, scores])
        
        with open(os.path.join(cfg.output_dir, "proposal_val.json"), "w") as f:
            json.dump(preds, f)

    def inference(
        self
    ) -> None:
        segment.info("2D detector inferencing...")
        self.config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0
        predictor = DefaultPredictor(self.config)

        preds = []
        dataset = self.datamodule.test_dataset()
        for dat in tqdm(dataset):
            im = np.array(Image.open(dat["file_name"]))
            out = predictor(im)
            boxes, scores, labels = out["instances"].pred_boxes, out["instances"].scores, out["instances"].pred_classes
            boxes = boxes.tensor
            keep_ids = batched_nms(boxes, scores, labels, iou_threshold=0.5)
            boxes, scores, labels = [
                arr[keep_ids].cpu().numpy() for arr in [boxes, scores, labels]
            ]
            keep_ids = np.argsort(scores)[::-1][:cfg.max_bbox]
            boxes, scores, labels = [
                arr[keep_ids].tolist() for arr in [boxes, scores, labels]
            ]
            preds.append([boxes, labels, scores])
        
        with open(os.path.join(cfg.output_dir, "proposal_test.json"), "w") as f:
            json.dump(preds, f)

    def visualize(
        self
    ) -> None:
        segment.info("2D detector visualizing...")
        bbox, label, score = [], [], []
        with open(os.path.join(cfg.output_dir, "proposal_test.json")) as f:
            data = json.load(f)
            for (box, lab, sco) in data:
                bbox.append(np.array(box))
                label.append(np.array(lab))
                score.append(np.array(sco))

        dataset = self.datamodule.test_dataset()
        os.makedirs(os.path.join(cfg.output_dir, "demo"), exist_ok=True)
        for dat, box in tqdm(list(zip(dataset, bbox))):
            img = np.array(Image.open(dat["file_name"]))
            img = demo(img, box)
            Image.fromarray(img).save(
                os.path.join(
                    cfg.output_dir,
                    "demo",
                    os.path.basename(dat["file_name"]).replace("_color_kinect", "")
                )
            )
