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

"""
archs = [
        Unet,
        UnetPlusPlus,
        MAnet,
        Linknet,
        FPN,
        PSPNet,
        DeepLabV3,
        DeepLabV3Plus,
        PAN,
    ]
"""

class SegNet(nn.Module):
    def __init__(self, arch='UnetPlusPlus', encoder_name="resnet101", in_channels=1, out_classes=3, encoder_weights="imagenet"):
        super(SegNet, self).__init__()
        self.model = smp.create_model(arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, encoder_weights=encoder_weights)
        self.lossfunc = DiceLoss(sigmoid=True)

    def forward(self, image_x, labels):
        output_x = self.model(image_x)
        loss = self.lossfunc(output_x, labels)
        loss_return = loss
        return {
            'loss': loss,
            'loss_return': loss_return,
            'logits': output_x,
            'labels': labels
        }
