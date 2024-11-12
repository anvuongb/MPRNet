"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import numpy as np
import os
import argparse
import torch.utils
import torch.utils.data
from tqdm import tqdm

import torch
import torch.nn as nn
import utils

from MPRNet import MPRNet
from skimage import img_as_ubyte
import scipy.io as sio
from pdb import set_trace as stx
from PIL import Image

from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid

parser = argparse.ArgumentParser(description="Image Denoising using MPRNet")

parser.add_argument(
    "--input_dir",
    type=str,
    help="Directory of validation images",
)
parser.add_argument(
    "--result_dir", type=str, help="Directory for results"
)
parser.add_argument(
    "--weights",
    default="./pretrained_models/model_denoising.pth",
    type=str,
    help="Path to weights",
)
parser.add_argument("--gpus", default="1", type=str, help="CUDA_VISIBLE_DEVICES")
parser.add_argument(
    "--save_images",
    action="store_true",
    help="Save denoised images in result directory",
)
parser.add_argument("--batch_size", type=int, default=4)

@torch.no_grad()
def save_image_2(
    tensor,
    fp,
    format=None,
    **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = (
        grid.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)

class CelebADataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_paths = os.listdir(img_dir)
        self.img_paths = [
            os.path.join(img_dir, f"{idx}.png") for idx in range(2096)
        ]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = read_image(img_path)
        image = image / 255.
        return image

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_restoration = MPRNet()

utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
# model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# Process data
data = CelebADataset(args.input_dir)
loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=6,
        drop_last=False,
    )
pbar = tqdm(enumerate(loader), total=len(loader))

# create output dir
if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

print(f"Generating denoised images from {args.input_dir}")
print(f"Images will be saved to {args.result_dir}")

for idx, batch in pbar:
    batch = batch.to("cuda:0")
    denoised = model_restoration(batch)
    for idx_, im in enumerate(denoised):
            num = idx * args.batch_size + idx_
            path = os.path.join(args.result_dir, f"{num}.png")
            save_image_2(im, path)
