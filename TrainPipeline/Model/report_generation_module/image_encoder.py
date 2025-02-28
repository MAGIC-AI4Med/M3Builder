import numpy as np
import sys

from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from monai.transforms import LoadImage, Compose, Resize, RandAffine, Rand2DElastic, Rand3DElastic, RandGaussianNoise, RandAdjustContrast
from safetensors.torch import load_model
from transformers import AutoModel,BertConfig,AutoTokenizer
import segmentation_models_pytorch as smp
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from .resnet2D import resnet101_2D, ResNet101_Weights

class ImageEncoder(nn.Module):
    def __init__(self, patch_num=44, fea_dim=2048, hidden_dim=1024):
        super(ImageEncoder, self).__init__()
        self.patch_num = patch_num
        self.image_encoder = resnet101_2D(weights=ResNet101_Weights.DEFAULT)
        self.mask_encoder = resnet101_2D(weights=ResNet101_Weights.DEFAULT)
        self.image_fc = nn.Sequential(
            nn.Linear(fea_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mask_fc = nn.Sequential(
            nn.Linear(fea_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pos_emb = nn.Parameter(torch.randn(1, patch_num, hidden_dim, dtype=torch.float32))
    
    def forward(self, image_x, mask_x):
        bs = mask_x.shape[0]
        image_x = image_x.repeat(1, 3, 1, 1)
        image_hidden_x = self.image_encoder(image_x)['x']
        image_hidden_x = self.image_fc(image_hidden_x)    
        image_hidden_x = image_hidden_x.unsqueeze(1)
        mask_x = rearrange(mask_x, 'b d h w -> (b d) h w')
        mask_x = mask_x.unsqueeze(1).repeat(1, 3, 1, 1)
        mask_hidden_x = self.mask_encoder(mask_x)['x']
        mask_hidden_x = self.mask_fc(mask_hidden_x)
        mask_hidden_x = rearrange(mask_hidden_x, '(b d) h -> b d h', b=bs)
        combined_features = torch.cat([image_hidden_x, mask_hidden_x], dim=1)
        output_x = combined_features + self.pos_emb
        return output_x

