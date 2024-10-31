from glob import glob
from tqdm import tqdm
import numpy as np
import os
from natsort import natsorted
import cv2
from joblib import Parallel, delayed
import multiprocessing
import argparse

parser = argparse.ArgumentParser(description='Generate patches from Full Resolution images')
parser.add_argument('--src_dir', default='../SIDD_Medium_Srgb/Data', type=str, help='Directory for full resolution images')
parser.add_argument('--tar_dir', default='../SIDD_patches/train',type=str, help='Directory for image patches')
# parser.add_argument('--ps', default=64, type=int, help='Image Patch Size')
parser.add_argument('--num_patches', default=10, type=int, help='Number of patches per image')
parser.add_argument('--num_cores', default=15, type=int, help='Number of CPU Cores')
parser.add_argument('--timesteps', type=int, required=True)
parser.add_argument('--type', type=str, required=True)

args = parser.parse_args()

src = args.src_dir
tar = args.tar_dir
# PS = args.ps
if args.type == "train":
    PS = 64
else:
    PS = 128
NUM_PATCHES = args.num_patches
NUM_CORES = args.num_cores

noisy_patchDir = os.path.join(tar, 'input')
clean_patchDir = os.path.join(tar, 'target')

if os.path.exists(tar):
    os.system("rm -r {}".format(tar))

os.makedirs(noisy_patchDir)
os.makedirs(clean_patchDir)

#get sorted folders
files = natsorted(glob(os.path.join(src,'*', '*.png')))

noisy_files, clean_files = [], []
for file_ in files:
    filename = os.path.split(file_)[-2]
    if 'clean' in filename and args.type in filename:
        clean_files.append(file_)
    if 'noisy' in filename and args.type in filename and str(args.timesteps) in filename:
        noisy_files.append(file_)
# print(clean_files[:100])
# print(noisy_files[:100])
print(f"Start processing {len(noisy_files)} files")

def save_files(i):
    noisy_file, clean_file = noisy_files[i], clean_files[i]
    noisy_img = cv2.imread(noisy_file)
    clean_img = cv2.imread(clean_file)

    H = noisy_img.shape[0]
    W = noisy_img.shape[1]
    for j in range(NUM_PATCHES):
        rr = np.random.randint(0, H - PS)
        cc = np.random.randint(0, W - PS)
        noisy_patch = noisy_img[rr:rr + PS, cc:cc + PS, :]
        clean_patch = clean_img[rr:rr + PS, cc:cc + PS, :]

        cv2.imwrite(os.path.join(noisy_patchDir, '{}_{}.png'.format(i+1,j+1)), noisy_patch)
        cv2.imwrite(os.path.join(clean_patchDir, '{}_{}.png'.format(i+1,j+1)), clean_patch)

Parallel(n_jobs=NUM_CORES)(delayed(save_files)(i) for i in tqdm(range(len(noisy_files))))
