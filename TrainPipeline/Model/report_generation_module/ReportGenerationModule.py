import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from transformers.generation.beam_search import BeamSearchScorer
import torchvision.models as models
from einops import rearrange

from .resAttention import Attention1D
from .language_model import LanguageModel

class ImageEncoder(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=1024, depth=120):
        super(ImageEncoder, self).__init__()

        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50_backbone = nn.Sequential(*list(self.resnet50.children())[:-1])
        # 添加一个新的全连接层，将输出维度修改为1024
        self.fc1 = nn.Linear(self.resnet50.fc.in_features, hidden_dim)

        # 自注意力机制：用于根据marks掩盖不重要的图像层
        # self.attention = nn.MultiheadAttention(embed_dim=144, num_heads=4, batch_first=True)
        self.attention1D = Attention1D(hid_dim=hidden_dim, max_depth=depth, nhead=1, num_layers=1, pool='cls', batch_first=True)
        # 最后的全连接层，压缩特征到hidden_dim
        # self.fc = nn.Linear(128 * 240 * 240, hidden_dim)

    def forward(self, image, marks):
        batch_size, channels, height, width, depth = image.shape
        
        image_x = image.repeat(1,3,1,1,1)
        image_x = rearrange(image_x, 'b c h w d -> (b d) c h w')
        
        x = self.resnet50_backbone(image_x) 
        x = torch.flatten(x, 1)
        x = self.fc1(x) # (b d), 2048
        
        output_x = rearrange(x, '(b d) h -> b d h', b=batch_size) # torch.Size([4, 120, 2048])
        output_x, _, _ = self.attention1D(output_x, marks) # torch.Size([4, 2048])
        
        return output_x


class MedImageReportModel(nn.Module):
    def __init__(self):
        super(MedImageReportModel, self).__init__()
        self.image_encoder = ImageEncoder()
        self.report_generator = LanguageModel(img_patch_num=1, max_tokens=1024)

    def forward(self, image, marks, input_ids, attention_mask, labels):
        # 图像编码
        encoded_features = self.image_encoder(image, marks)

        loss = self.report_generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_hidden_states=encoded_features,
            return_loss=True
        )

        if self.training:
            return {"loss": loss}
        else:
            output = self.report_generator.generate(
                image_hidden_states=encoded_features,
                max_length=512,
                num_beams=1,
                num_beam_groups=1,
                do_sample=False,
                num_return_sequences=1,
                early_stopping=True
            )
            return {
                'loss': loss,
                'loss_return': loss,
                'output': output,
                'labels': labels
            }