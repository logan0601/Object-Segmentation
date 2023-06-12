import os
import glob
import json
import pickle
import logging
from PIL import Image

import torch
import numpy as np
from tqdm import tqdm


import segment
from segment.utils.config import cfg
from segment.utils.transform import estimate_poses
from segment.utils.typing import *


def process(
    file_list: List[str],
    proc_dir: str,
    test_set: bool
) -> None:
    segment.info(
        f"Processing data {os.path.dirname(file_list[0])} of size {len(file_list)}")

    os.makedirs(proc_dir, exist_ok=True)
    if not test_set:
        for i, fn in tqdm(list(enumerate(file_list))):
            fn2 = fn[:-17] + "_depth_kinect.png"
            fn3 = fn[:-17] + "_label_kinect.png"
            fn4 = fn[:-17] + "_meta.pkl"

            name = os.path.basename(fn)[:-17]
            meta_p = os.path.join(proc_dir, name + "_meta.json")
            data_p = os.path.join(proc_dir, name + "_data.pth")

            rgb = np.array(Image.open(fn))
            dep = np.array(Image.open(fn2))
            lab = np.array(Image.open(fn3))
            intrin = pickle.load(open(fn4, "rb"))["intrinsic"]

            # bbox
            box = []
            sem = np.unique(lab)
            sem = [i for i in sem if i < cfg.num_objects]
            for sem_lab in sem:
                y, x = np.nonzero(lab == sem_lab)
                box.append([int(y.min()), int(x.min()), int(y.max()), int(x.max()), sem_lab])
            
            with open(meta_p, "w") as f:
                json.dump({
                    "file_name": fn,
                    "image_id": i,
                    "height": rgb.shape[0],
                    "width": rgb.shape[1],
                    "annotations": [
                        {
                            "bbox": bbox[:-1],
                            "bbox_mode": 0,
                            "category_id": int(bbox[-1])
                        }
                        for bbox in box
                    ]
                }, f)
            
            torch.save((
                torch.as_tensor(rgb, dtype=torch.uint8),
                torch.as_tensor(dep, dtype=torch.uint8),
                torch.as_tensor(lab, dtype=torch.uint8),
                torch.as_tensor(intrin, dtype=torch.float32),
                torch.as_tensor(box, dtype=torch.int32)
            ), data_p)
    
    else:
        for i, fn in tqdm(list(enumerate(file_list))):
            fn2 = fn[:-17] + "_depth_kinect.png"
            fn3 = fn[:-17] + "_meta.pkl"

            name = os.path.basename(fn)[:-17]
            meta_p = os.path.join(proc_dir, name + "_meta.json")
            data_p = os.path.join(proc_dir, name + "_data.pth")

            rgb = np.array(Image.open(fn))
            dep = np.array(Image.open(fn2))
            intrin = pickle.load(open(fn3, "rb"))["intrinsic"]

            with open(meta_p, "w") as f:
                json.dump({
                    "file_name": fn,
                    "image_id": i,
                    "height": rgb.shape[0],
                    "width": rgb.shape[1]
                }, f)
            
            torch.save((
                torch.as_tensor(rgb, dtype=torch.uint8),
                torch.as_tensor(dep, dtype=torch.uint8),
                torch.as_tensor(intrin, dtype=torch.float32)
            ), data_p)


def process_pose(
    file_list: List[str],
    proc_dir: str,
) -> None:
    segment.info("Processing pose...")

    os.makedirs(proc_dir, exist_ok=True)
    for i, fn in tqdm(list(enumerate(file_list))):
        meta = pickle.load(open(fn, "rb"))
        pose_world = np.array([meta["poses_world"][idx] for idx in meta["object_ids"]])
        box_sizes = np.array([meta["extents"][idx] * meta["scales"][idx] for idx in meta["object_ids"]])
        corners, poses = [], []
        for i in range(len(pose_world)):
            corner, pose = estimate_poses(
                center=pose_world[i][:3, 3],
                size=box_sizes[i],
                rotation=pose_world[i][:3, :3],
                extrinsic=meta["extrinsic"]
            )
            corners.append(corner)
            poses.append(pose)
        
        corners = torch.from_numpy(np.stack(corners, axis=0, dtype=np.float32))
        poses = torch.from_numpy(np.stack(poses, axis=0, dtype=np.float32))
        torch.save((
            corners,
            poses
        ), os.path.join(proc_dir, os.path.basename(fn)[:-4] + ".pth"))


def setup():
    # instaniate a logger
    logger = logging.getLogger("object_segmentation")
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    log_format = '%(asctime)s | %(levelname)s: %(message)s'
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    # process train/val data
    proc_dir = os.path.join(os.path.dirname(cfg.train_dir), "process")
    if not os.path.exists(proc_dir) \
        or len(os.listdir(proc_dir)) == 0:
        train_list = sorted(glob.glob(os.path.join(cfg.train_dir, "*_color_kinect.png")))
        process(train_list, proc_dir, test_set=False)
    
    # process test data
    proc_dir = os.path.join(os.path.dirname(cfg.test_dir), "process")
    if not os.path.exists(proc_dir) \
        or len(os.listdir(proc_dir)) == 0:
        test_list = sorted(glob.glob(os.path.join(cfg.test_dir, "*_color_kinect.png")))
        process(test_list, proc_dir, test_set=True)

    proc_dir = os.path.join(os.path.dirname(cfg.applic_dir), "process")
    if not os.path.exists(proc_dir) \
        or len(os.listdir(proc_dir)) == 0:
        appl_list = sorted(glob.glob(os.path.join(cfg.applic_dir, "*_meta.pkl")))
        process_pose(appl_list, proc_dir)
