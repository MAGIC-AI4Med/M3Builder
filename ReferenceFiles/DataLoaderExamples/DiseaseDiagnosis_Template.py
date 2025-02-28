import json
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import AutoModel
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
import re
import pickle
from PIL import Image
random.seed(42)

# You must not modify or delete this function
def resample_image(image, tS, tD):
    depth = image.shape[0]
    height = image.shape[-2]
    width = image.shape[-1]
    # assert height == width
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

# You would better not modify content of this class
class DiseaseDiagnosisTemp_dataloader(Dataset):
    # You must not modify this init function
    def __init__(self, data_path, label_path, num_classes=4, size=256, depth=32, partial=1):
        with open(data_path, 'r') as f:
            self.data_list = json.load(f)
        with open(label_path, 'r') as f:
            self.label_dict = json.load(f)
        self.cls = num_classes
        self.size = size
        self.depth = depth
        self.datas = list(self.data_list.keys())
        length = len(self.datas)
        self.datas = self.datas[:int(length*partial)]
        self.partial = partial
    
    def __getitem__(self, index):
        path = self.datas[index] # You Must not modify this line
        image_fuse = torch.zeros((1,self.size,self.size,self.depth), dtype=torch.float32)
        mark = torch.zeros(self.depth)
        label_key_list = self.data_list[path]
        label_key_list = list(set(label_key_list))
        loc_list = [self.label_dict[ele] for ele in label_key_list if ele in self.label_dict]
        labels = torch.zeros(self.cls)
        labels[loc_list] = 1
        labels = labels.to(torch.float32)

        #process image
        if path.endswith('jpg') or path.endswith('png'):
            image_path = path
            image_datas = np.array(Image.open(image_path).convert('L')) # You may change the way to load the image according to its file type, such as .pt, .nii.gz

        elif path.endswith('.pt') or path.endswith('.nii.gz') or path.endswith('.nii'):
            image_path = path
            image_datas = "please load the image from image_path into numpy array, accoring to file format"
            
        # This part must not be modified!!!
        image_tensor = torch.tensor(image_datas, dtype=torch.float32) # You Must not modify this line
        if len(image_tensor.shape)==3: # You Must not modify this line
            image_tensor = image_tensor.unsqueeze(0) # You Must not modify this line
        if len(image_tensor.shape)==2: # You Must not modify this line
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(-1) # You Must not modify this line
        assert len(image_tensor.shape)==4 # You Must not modify this line

        image_tensor = rearrange(image_tensor, "c h w d -> d c h w") # You Must not modify this line
        image_tensor = resample_image(image_tensor, self.size, self.depth) # You Must not modify this line
        image_tensor = rearrange(image_tensor, "d c h w -> c h w d") # You Must not modify this line
        
        # This part must not be modified!!!
        mark[:image_tensor.shape[-1]] = 1
        if image_tensor.max()-image_tensor.min() == 0:
            image_tensor = torch.randn_like(image_tensor, dtype=torch.float32)
        image = (image_tensor - image_tensor.min()) / (image_tensor.max()-image_tensor.min())
        image_fuse[:,:,:,0:image.shape[-1]] = image
        
        assert image_fuse.shape[-1] == mark.shape[0]
        # You must return these 3 entries !!!!!!
        return {
            "images": image_fuse,
            "marks": mark,
            "labels": labels,
            }
        
    
    def __len__(self):
        return len(self.datas)
    
if __name__ == "__main__":
   
    datasets = DiseaseDiagnosisTemp_dataloader('path/to/train.json',
                        "path/to/label_dict.json",  
                        )
    dataloader = DataLoader(
            datasets,
            batch_size=4,
            num_workers=16,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=None,
            drop_last=True,
        )  
    for i, sample in enumerate(dataloader):
        pass # Here dont use print due to the context length, you just need to recieve error message if there is any error