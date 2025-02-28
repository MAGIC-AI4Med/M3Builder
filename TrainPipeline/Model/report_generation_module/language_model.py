from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torchinfo import summary
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from transformers.generation.beam_search import BeamSearchScorer

import json

class LanguageModel(nn.Module):
    """
    GPT2 model with a language modeling head and pseudo self-attention.
    """

    def __init__(self, img_patch_num, max_tokens=1024):
        super().__init__()
        self.bos_token_id = 50256
        self.eos_token_id = 50256
        self.pad_token_id = 50256

        # Initialize GPT2 with config directly
        config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=1024,
            n_layer=24,
            n_head=16,
            n_inner=4096,
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            summary_type="cls_index",
            summary_use_proj=True,
            summary_activation=None,
            summary_proj_to_labels=True,
            summary_first_dropout=0.1,
            scale_attn_weights=True,
            use_cache=True,
            bos_token_id=50256,
            eos_token_id=50256,
        )
        
        self.gpt_with_lm_head = GPT2LMHeadModel(config)
        
        self.max_tokens = max_tokens
        self.img_patch_num = img_patch_num

    def forward(self,
                input_ids,  # shape [batch_size x seq_len]
                attention_mask,  # shape [batch_size x seq_len]
                image_hidden_states,  # shape [batch_size x word_hidden_dim]
                return_loss: bool = False,
                use_cache: bool = False,
                labels=None
                ):
        """
        If return_loss is True, returns the language modeling loss.
        If return_loss is False (in which we are in text generation mode and use_cache will be True), returns the language modeling logits.
        """
        # For training
        if return_loss:
            outputs = self.gpt_with_lm_head(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,  # Use input_ids as labels for causal LM
                use_cache=use_cache,
            )
            loss = outputs.loss
            return loss
        # For generation/inference
        else:
            outputs = self.gpt_with_lm_head(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
            )
            return outputs.logits, outputs.past_key_values

    @torch.no_grad()
    def generate(self,
                 image_hidden_states: torch.FloatTensor,  # shape [batch_size x image_hidden_dim]
                 max_length: int = None,
                 num_beams: int = 1,
                 num_beam_groups: int = 1,
                 do_sample: bool = False,
                 num_return_sequences: int = 1,
                 early_stopping: bool = False
                 ) -> torch.LongTensor:  # shape [batch_size x longest_generated_sequence_length]

        batch_size = image_hidden_states.size(0)
        device = image_hidden_states.device
        input_ids = torch.full(size=(batch_size, 1), fill_value=self.bos_token_id, dtype=torch.int64, device=device)
        attention_mask = torch.ones(size=(batch_size, 1), dtype=torch.int64, device=device)

        outputs = self.gpt_with_lm_head.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length if max_length is not None else 512,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            early_stopping=early_stopping,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )

        return outputs