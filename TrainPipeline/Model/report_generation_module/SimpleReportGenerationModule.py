import torch
import torch.nn as nn
import torchvision.models as models
from einops import rearrange

class ImageEncoder(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=1024, depth=120):
        super(ImageEncoder, self).__init__()
        
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50_backbone = nn.Sequential(*list(self.resnet50.children())[:-1])
        self.fc1 = nn.Linear(self.resnet50.fc.in_features, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        
    def forward(self, image, marks):
        batch_size, channels, height, width, depth = image.shape
        
        image_x = image.repeat(1,3,1,1,1)
        image_x = rearrange(image_x, 'b c h w d -> (b d) c h w')
        
        x = self.resnet50_backbone(image_x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        
        output_x = rearrange(x, '(b d) h -> b d h', b=batch_size)
        output_x = output_x * marks.unsqueeze(-1)
        output_x, _ = self.attention(output_x, output_x, output_x)
        output_x = output_x.mean(dim=1)
        
        return output_x

class SimpleDecoder(nn.Module):
    def __init__(self, hidden_dim=1024, vocab_size=1000):
        super(SimpleDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, 512, hidden_dim))
        
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=4
        )
        
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, image_features, input_ids, attention_mask):
        # Create embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding[:, :input_ids.size(1), :]
        embeddings = token_embeddings + position_embeddings
        
        # Create attention mask for transformer
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Expand image features for transformer
        image_features = image_features.unsqueeze(1)
        
        # Pass through transformer
        decoder_output = self.transformer(
            embeddings,
            image_features,
            tgt_mask=self.generate_square_subsequent_mask(input_ids.size(1)).to(input_ids.device),
            tgt_key_padding_mask=~attention_mask.squeeze(1).squeeze(1).bool()
        )
        
        # Generate output logits
        logits = self.output_layer(decoder_output)
        
        return logits
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class SimpleReportGenerationModel(nn.Module):
    def __init__(self, vocab_size=1000):
        super(SimpleReportGenerationModel, self).__init__()
        self.image_encoder = ImageEncoder()
        self.decoder = SimpleDecoder(vocab_size=vocab_size)
        
    def forward(self, image, marks, input_ids, attention_mask, labels=None):
        # Encode image
        encoded_features = self.image_encoder(image, marks)
        
        # Generate text
        logits = self.decoder(encoded_features, input_ids, attention_mask)
        
        if labels is not None:
            # Training mode
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.decoder.vocab_size), labels.view(-1))
            return {
                'loss': loss,
                'loss_return': loss,
                'output': input_ids,
                'labels': labels
            }
        else:
            # Inference mode
            return {
                'loss': torch.tensor(0.0),
                'loss_return': torch.tensor(0.0),
                'output': logits.argmax(-1),
                'labels': input_ids
            }