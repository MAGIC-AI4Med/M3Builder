import numpy as np
import sys
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from monai.transforms import LoadImage, Compose, Resize, RandAffine, Rand2DElastic, Rand3DElastic, RandGaussianNoise, RandAdjustContrast
from safetensors.torch import load_model
from transformers import AutoModel,BertConfig,AutoTokenizer

from .resnet2D import resnet50_2D, ResNet50_Weights, resnet34_2D, ResNet34_Weights
from .vitFuse import vitFuse
from .resAttention import Attention1D
from .Utils.utils import visual_augment
from Loss.AllLosses import  MultiLabelLoss, MSEwithGaussian, KLwithGaussian, SoftCrossEntropy, InfoNCELoss


rand_Gaussian_3d = RandGaussianNoise(
    prob=0.15,
    mean=0,
    std=0.07
)
rand_contrast_3d = RandAdjustContrast(
    prob=0.15,
    gamma=(0.5,1.5)
)

rand_affine = RandAffine(
    prob=0.15,
    rotate_range=(0, 0, np.pi/12),  # 以弧度为单位，这里仅在Z轴方向上旋转
    translate_range=(10, 10, 0),  # 在每个轴向上平移的像素范围
    scale_range=(0.1, 0.1, 0),  # 缩放比例
    shear_range=(0.2, 0.2, 0),  # 剪切强度
    mode='bilinear',  # 插值方式
    padding_mode='zeros'  # 填充方式
)

rand_3d_elastic = Rand3DElastic(
    prob=0.15,  # 50% 概率应用变换
    sigma_range=(5,7),
    magnitude_range=(10, 20),  # 变形的幅度范围
    # spacing=(20, 20),  # 控制位移网格的间距
    rotate_range=(0, 0, 0),  # 旋转范围，这里没有旋转
    scale_range=(0.1, 0.1, 0.1),  # 缩放范围
    mode='bilinear',  # 插值模式
    padding_mode='border'  # 边界填充模式
)

transform = Compose([
    rand_affine,
    rand_3d_elastic,
    rand_Gaussian_3d,
    rand_contrast_3d,
])


class RadNet(nn.Module):
    def __init__(self, num_cls=5569, backbone='resnet', size=256, depth=32, hid_dim=2048, ltype='MultiLabel', augment=False):
        super(RadNet, self).__init__()
        self.size = size
        self.depth = depth
        self.hid_dim = hid_dim
        self.pos_weights = torch.ones([num_cls]) * 20 #300, 100, 10000, 80, 50
        self.cls = num_cls
        self.backbone = backbone
        if self.backbone == 'resnet':
            # self.resnet2D = resnet50_2D(weights=("pretrained", ResNet50_Weights.DEFAULT))
            self.resnet2D = resnet50_2D()
            # self.resnet2D = resnet34_2D(weights=("pretrained", ResNet34_Weights.DEFAULT))
            self.attention1D = Attention1D(hid_dim=hid_dim, max_depth=depth, nhead=1, num_layers=1, pool='cls', batch_first=True)


        self.vitFuse = vitFuse(mod_num=16, hid_dim=2048, nhead=4, num_layers=2, pool='cls', batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, self.cls)
        )
        
        self.augment = augment
        self.ltype = ltype
        self.lossfunc = None
        if self.ltype == 'MultiLabel':
            self.lossfunc = MultiLabelLoss()
        else:
            raise ValueError("Invalid Loss Function")


    def forward(self, image_x, marks, labels):
        tmp_Aug = False
        labels = labels.squeeze()
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(0)
        if self.training and self.augment:
            tmp_Aug = True

        B = image_x.shape[0]
        image_x = image_x.repeat(1,3,1,1,1)

        image_x = rearrange(image_x, 'b c h w d -> (b d) c h w')
        
        if tmp_Aug:
            image_x = rearrange(image_x, "b c h w -> b h w c")
            image_x = transform(image_x)
            image_x = rearrange(image_x, "b h w c -> b c h w")
        if self.backbone == 'resnet':
            output = self.resnet2D(image_x)
            output_x = output["x"]
            
        output_x = rearrange(output_x, '(b d) h -> b d h', b=B)
        output_x, _, _ = self.attention1D(output_x, marks)
        output_x = self.fc(output_x)
        loss_cls = self.lossfunc(output_x, labels)
        return {
            'loss': loss_cls,
            'loss_return': loss_cls,
            'logits': output_x,
            'labels': labels,
        }


            
    



            
    

