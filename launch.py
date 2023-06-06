import torch
import numpy as np
from PIL import Image

from objseg.models.rcnn import FasterRCNN
from objseg.utils.ops import scale_tensor


# net = FasterRCNN(80)

# path = "datasets/example_data/1-1-0_color_kinect.png"
# im = torch.from_numpy(np.array(Image.open(path), dtype=np.float32) / 255)
# im = im.permute(2, 0, 1).unsqueeze(0)

# feats = net(im)
# feats = feats[0].permute(1, 2, 0)
# feats = scale_tensor(feats, [feats.min(), feats.max()], [0, 1])

# feats = feats.detach().cpu().numpy() * 255.0
# print(feats.shape)

# im = Image.fromarray(feats.astype(np.float32))
# im.save("datasets/example_data/3.png")


import glob
import pickle
from tqdm import tqdm


dataset = "../datasets/training_data/data"
tot = len(glob.glob(f"{dataset}/*.pkl"))

for i, path in enumerate(glob.glob(f"{dataset}/*.pkl")):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        assert len(data.keys()) == 1
    
    if (i + 1) % 2000 == 0:
        print(f"{i + 1}/{tot}")
