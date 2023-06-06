import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2


edge = [[0, 1], [1, 2], [2, 3], [3, 0]]


def draw(image: np.ndarray, uv: np.ndarray):
    for (i, j) in edge:
        cv2.line(
            image,
            (uv[i, 0], uv[i, 1]),
            (uv[j, 0], uv[j, 1]),
            color=(0, 1, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return image


def demo(name: str):
    datasets = "../datasets/training_data"
    image = np.array(Image.open(f"{datasets}/data/{name}_color_kinect.png"))
    bbox = np.load(f"{datasets}/preprocess/{name}_bbox.npz")["gt_boxes"].astype(np.int32)
    for i in range(bbox.shape[0]):
        y_lo, x_lo, y_hi, x_hi = bbox[i][:4]
        if x_lo == x_hi or y_lo == y_hi:
            break
        box = np.array([
            [x_lo, y_lo],
            [x_hi, y_lo],
            [x_hi, y_hi],
            [x_lo, y_hi]
        ])
        image = draw(image, box)

    Image.fromarray(image).save(f"../datasets/example_data/example.png")


def search_bbox(label: torch.Tensor) -> torch.Tensor:
    NUM_OBJECTS = 79
    objects = torch.arange(NUM_OBJECTS, dtype=torch.int32).view(-1)[:, None, None].cuda()

    res = (objects == label[None]).int()
    ids = torch.nonzero(res.sum(dim=[1, 2])).view(-1)

    res = res[ids]
    bbox = []
    for i in range(res.shape[0]):
        yx = torch.nonzero(res[i])
        y_lo, y_hi = yx[:, 0].min(), yx[:, 0].max()
        x_lo, x_hi = yx[:, 1].min(), yx[:, 1].max()
        bbox.append(torch.stack([y_lo, x_lo, y_hi, x_hi, ids[i]]))

    bbox = torch.stack(bbox, dim=0)
    return bbox


def main():
    datasets = "../datasets/training_data/data"
    bboxes = []
    min_obj, max_obj = 10000, 0
    label_list = [path for path in os.listdir(f"{datasets}") if "_label_kinect.png" in path]
    print(len(label_list))
    for path in tqdm(label_list):
        label = torch.from_numpy(np.array(Image.open(f"{datasets}/{path}"), dtype=np.int32)).cuda()
        bbox = search_bbox(label)
        min_obj = min(min_obj, bbox.shape[0])
        max_obj = max(max_obj, bbox.shape[0])
        # assert list(bbox.shape) == [5, 5], f"{path} has {bbox.shape}"
        bboxes.append(bbox)
    
    print(min_obj, max_obj)

    output = torch.zeros(len(bboxes), max_obj, 5)
    for i, bbox in enumerate(bboxes):
        output[i][:bbox.shape[0]] = bbox
    
    print(output.shape)
    output = output.detach().cpu().numpy()
    np.savez("../datasets/training_data/gt2D.npz", gt_boxes=output)

    dst = "../datasets/training_data/preprocess"
    for i, (path, bbox) in enumerate(zip(label_list, bboxes)):
        np.savez(
            f"{dst}/{os.path.basename(path).replace('_label_kinect.png', '_bbox.npz')}",
            gt_boxes=output[i],
            num_boxes=bbox.shape[0]
        )


if __name__ == "__main__":
    demo("1-1-20")
    # main()
