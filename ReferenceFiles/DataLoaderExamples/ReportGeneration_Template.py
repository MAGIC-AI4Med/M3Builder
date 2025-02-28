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


def Normalization(torch_image):
    np_image = torch_image.numpy()
    lower_bound, upper_bound = np.percentile(np_image, 0.5), np.percentile(np_image, 99.5)
    np_image = np.clip(np_image, lower_bound, upper_bound)
    np_image = (np_image - np.mean(np_image)) / np.std(np_image)
    return torch.tensor(np_image)


def resample_image(image, tS, tD):
    depth = image.shape[0]
    height = image.shape[-2]
    width = image.shape[-1]
    size = height

    if size == tS and width == tS and depth <= tD:
        output_tensor = image
    else:
        output_tensor = torch.nn.functional.interpolate(image, size=(tS, tS), mode='bilinear', align_corners=False)
        if depth > tD:
            step = depth / tD
            indices = indices = torch.arange(0, depth, step).long()
            indices = indices[:tD]
            output_tensor = output_tensor[indices]
    return output_tensor


class ReportGenerationTemp_dataloader(Dataset):
    """
    Dataset for loading 2D & 3D images and associated findings and impressions, ensuring consistent output formats.

    Key points:
    - Output Format: Both 2D and 3D inputs are processed into a consistent output shape:
    - Image tensor: (1, size, size, depth)
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
                 image_size=(240,240), # you must not modify this parameter!
                 depth=120, # you must not modify this parameter! It should be exactly 120!
                 partial=1,
                 ): # you must not modify this parameter!
        self.datas = self.load_data(json_path)

        self.dimensionality = "Here you must assign a default value, '2D' or '3D', according to dataset description!!!"
        assert self.dimensionality in ['2D', '3D']
    
        self.image_size = image_size
        self.depth = depth
        length = len(self.datas)
        self.datas = self.datas[:int(length*partial)]
        self.partial = partial

        self.tokenizer = GPT2Tokenizer.from_pretrained("healx/gpt-2-pubmed-medium")
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_data(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as file:
            datas = json.load(file)
            return datas
    
    # you must not modify this function
    def __getitem__(self, idx):
        sample = self.datas[idx]
        img_path = sample["image_path"]
        findings = sample["findings"]
        impressions = sample["impression"]

        mark = torch.zeros(self.depth)
        
        if self.dimensionality == '2D':
            image = Image.open(img_path).convert("L")
            if self.image_size is not None:
                image = image.resize(self.image_size)
            image = np.array(image)
            image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max()-image_tensor.min())
        elif self.dimensionality == '3D':
            nii_img = nib.load(img_path)
            img_data = nii_img.get_fdata()
            image_tensor = torch.from_numpy(img_data)
            image_tensor = Normalization(image_tensor)
        else:
            raise ValueError(f"Dimensionality must be '2D' or '3D'.")
        
        image_fuse = torch.zeros((1,self.image_size[0],self.image_size[1],self.depth), dtype=torch.float32)

        image_tensor = rearrange(image_tensor, "c h w d -> d c h w")
        image_tensor = resample_image(image_tensor, self.image_size[0], self.depth)
        image_tensor = rearrange(image_tensor, "d c h w -> c h w d")
        
        mark[:image_tensor.shape[-1]] = 1
        image_fuse[:,:,:,0:image_tensor.shape[-1]] = image_tensor

        tokenized_text = self.prepare_text(findings, impressions)

        # You must return these 4 entries:
        return {
            "images": image_fuse, # 1,240,240,120
            "marks": mark, # 120
            "input_ids": tokenized_text['input_ids'],
            "attention_mask": tokenized_text['attention_mask'],
        }

    def __len__(self):
        return len(self.datas)
    
    def prepare_text(self, findings, impressions):
        bos_token = "xxx" # Here Please fill in a bos_token you like
        eos_token = bos_token
        rep_with_special_tokens = bos_token + findings + eos_token
        inputs = self.tokenizer(rep_with_special_tokens, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze()
        }
    

if __name__ == "__main__":
   
    datasets = ReportGenerationTemp_dataloader(
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