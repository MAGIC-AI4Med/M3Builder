import json
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import monai
import numpy as np
import random
import nibabel as nib
import scipy
import ruamel.yaml as yaml
from tqdm import tqdm
import sys
import random
from einops import rearrange
import os
import csv
import re
import pickle
from PIL import Image
random.seed(42)

# you must not modify this function
def Normalization(torch_image):
    np_image = torch_image.numpy()
    lower_bound, upper_bound = np.percentile(np_image, 0.5), np.percentile(np_image, 99.5)
    np_image = np.clip(np_image, lower_bound, upper_bound)
    np_image = (np_image - np.mean(np_image)) / np.std(np_image)
    return torch.tensor(np_image)

# you must not modify this function
def Padding(torch_image, target_shape = (480,480,240), is_mask=False):
    dh, dw, dd = target_shape
    h, w, d = torch_image.shape
    if is_mask:
        pad_val = min([torch_image[0,0,0], torch_image[-1,0,0], torch_image[0,-1,0], 
                torch_image[0,0,-1], torch_image[0,-1,-1], torch_image[-1,0,-1], 
                torch_image[-1,-1,0], torch_image[-1,-1,-1]]).item()
    else:
        pad_val=0

    h_start = max((h - dh) // 2, 0)
    h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end = min(d_start + dd, d)

    tensor = torch_image[h_start:h_end, w_start:w_end, d_start:d_end]

    pad_h_before = (dh - tensor.shape[0]) // 2
    pad_h_after = dh - tensor.shape[0]- pad_h_before

    pad_w_before = (dw - tensor.shape[1]) // 2
    pad_w_after = dw - tensor.shape[1] - pad_w_before

    pad_d_before = (dd - tensor.shape[2]) // 2
    pad_d_after = dd - tensor.shape[2] - pad_d_before
    tensor = torch.nn.functional.pad(tensor,
        (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after),
        value=pad_val
    )

    return tensor

# Do not modify the content of this class
class OrganSegmentationTemp_dataloader(Dataset):
    """
    Dataset for loading 3D images and associated organ_masks and labels, ensuring consistent output formats.

    Key points:
    - Output Format: 3D inputs are processed into a consistent output shape:
    - Image tensor: (size, size, depth)
    - Mark tensor: (depth) where, for 2D images, the remaining (depth - 1) slices are set as a mask.
    - Depth Handling:
    - For 2D images, the image is padded along the depth dimension with mask indicators.
    - For 3D images or concatenated 2D images, only a maximum of 'depth' slices is considered.
    - The reample_image function is responsible for adjusting the depth dimension appropriately.
    - Image Loading:
    - The current implementation uses PIL to load images and converts them to grayscale.
    - If needed, adapt the loading method for different file formats (e.g., .pck via pickle or .nii.gz via nibabel).
    
    This template ensures that regardless of the input (2D or 3D), the output tensors maintain the same shape,
    with the required handling of the depth dimension via the reample_image function.
    """
    def __init__(self,
                 json_path, 
                 num_classes=1, 
                 image_size=(480,480,240), 
                 partial=1):
        self.datas = self.load_data(json_path)

        self.image_size = image_size
        length = len(self.datas)
        self.datas = self.datas[:int(length*partial)]
        self.partial = partial
    
    def load_data(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as file:
            datas = json.load(file)
            return datas

    def __getitem__(self, idx):
        sample = self.datas[idx]

        image_path = sample["image_path"]
        mask_path = sample["label_path"]
        labels = sample["labels"]

        nii_img = nib.load(image_path)
        img_data = nii_img.get_fdata()
        image_tensor = torch.from_numpy(img_data)
        image_tensor = Normalization(image_tensor)

        nii_mask = nib.load(mask_path)
        mask_data = nii_mask.get_fdata()
        mask_tensor = torch.from_numpy(mask_data)

        # You have to return these 3 entries:
        return {
            "images": image_tensor, # 480,480,240
            "organs_mask": mask_tensor,  # N,480,480,240
            "labels":labels
        }

    def __len__(self):
        return len(self.datas)
    

if __name__ == "__main__":
    # only validate train set, dont care about test set
    datasets = OrganSegmentationTemp_dataloader(
        "path/to/train.json"
    )
    dataloader = DataLoader(
            datasets,
            batch_size=1,
            num_workers=16,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=None,
            drop_last=True,
        )  
    for i, sample in enumerate(dataloader):
        pass # Here dont use print due to the context length, you just need to recieve error message if there is any error