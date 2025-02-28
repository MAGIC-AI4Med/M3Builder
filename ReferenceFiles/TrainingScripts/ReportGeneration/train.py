from tqdm import tqdm
import torch.nn.functional as F
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import transformers
import evaluate
from sklearn import metrics
from dataclasses import dataclass, field

import warnings
warnings.filterwarnings("ignore")

# Here Please import the best matching dataset class
from Datapath.dataloader import "Here Please import the best matching dataset class"

from Model.report_generation_module.RGModule import MedImageReportModel

import numpy as np
import torch
torch.set_printoptions(profile="full")

from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint, is_main_process


import random
import os
import matplotlib.pyplot as plt
from safetensors.torch import load_model

from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
import nltk # you dont need to download anything else!
from nltk.translate.bleu_score import sentence_bleu


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_file(directory, tg):
    files = os.listdir(directory)
    files = [file for file in files if file.startswith(tg)]
    assert len(files)==1
    return files[0]

@dataclass
class ModelArguments:
    tokenizer_name: Optional[str] = field(
        default='malteos/PubMedNCL', metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

@dataclass
class DataArguments:
    Mode: Optional[str] = field(default="Train")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default = False)
    per_device_train_batch_size: int = field(default = 4)
    per_device_eval_batch_size: int = field(default = 2)
    output_dir: Optional[str] = field(default="/")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")

    # My new parameters
    # patch_num: int = field(default=44)  
    # fea_dim: int = field(default=2048)
    # hidden_dim: int = field(default=1024)
    # size: int = field(default=256)
    checkpoint: Optional[str] = field(default=None)
    safetensor: Optional[str] = field(default=None)

@dataclass
class DataCollator(object):

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        images, marks, inputs, attns = tuple([instance[key] for instance in instances] for key in ('images', 'marks', 'input_ids', 'attention_mask'))
        images = torch.cat([_.unsqueeze(0) for _ in images],dim  = 0)
        marks = torch.cat([_.unsqueeze(0) for _ in marks],dim  = 0)
        inputs = torch.cat([_.unsqueeze(0) for _ in inputs],dim  = 0)
        attns = torch.cat([_.unsqueeze(0) for _ in attns],dim  = 0)
        return_dic = dict(
            image=images,
            marks=marks,
            input_ids=inputs,
            attention_mask=attns,
            labels=inputs
        )
        return return_dic
    
class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.eval_epoch = 0

    def on_evaluate(self, args, state, control, **kwargs):
        # 在这里保存评估信息
        self.eval_epoch = state.epoch
    
metrics_callback = MetricsCallback()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dice_score(logits, labels):
    scores = sigmoid(logits)
    scores_flat = scores.flatten()
    labels_flat = labels.flatten()
    intersection = np.sum(scores_flat * labels_flat)
    return (2. * intersection) / (np.sum(labels_flat) + np.sum(scores_flat))

tokenizer = GPT2Tokenizer.from_pretrained("healx/gpt-2-pubmed-medium")
tokenizer.pad_token = tokenizer.eos_token
def compute_metrics(eval_preds):
    # print(eval_preds)
    epoch = metrics_callback.eval_epoch
    predictions = eval_preds.predictions
    label_ids = eval_preds.label_ids
    loss,output,labels = predictions
    # print("Loss:",loss)
    # print("output:",type(output),output,output.shape) # 399, 512
    # print("Labels:",type(labels),labels,labels.shape) # 399, 512
    output = torch.tensor(output)
    labels = torch.tensor(labels)

    invalid_token_id = -100  # 假设 -1 是无效的 token id
    output[output == invalid_token_id] = 50256

    # print("output:", output)
    # print("labels:", labels)

    report_ref = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    report_gene = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    bleu_1_list = []
    bleu_2_list = []
    bleu_3_list = []
    bleu_4_list = []
    for gene, ref in tqdm(zip(report_gene, report_ref), total=len(report_gene)):
        tokenized_gene = nltk.word_tokenize(gene)
        tokenized_ref = nltk.word_tokenize(ref)
        bleu_1_list.append(sentence_bleu([tokenized_ref], tokenized_gene, weights=(1, 0, 0, 0)))
        bleu_2_list.append(sentence_bleu([tokenized_ref], tokenized_gene, weights=(0.5, 0.5, 0, 0)))
        bleu_3_list.append(sentence_bleu([tokenized_ref], tokenized_gene, weights=(0.33, 0.33, 0.33, 0)))
        bleu_4_list.append(sentence_bleu([tokenized_ref], tokenized_gene, weights=(0.25, 0.25, 0.25, 0.25)))
    
    metrics_return = {
        "loss": np.mean(loss),
        "bleu_1": np.mean(bleu_1_list),
        "bleu_2": np.mean(bleu_2_list),
        "bleu_3": np.mean(bleu_3_list),
        "bleu_4": np.mean(bleu_4_list)
    }

    return metrics_return
    
def main():
    set_seed()
    # print("cp1")
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # print("cp2")
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # print("cp3")

    checkpoint = None if training_args.checkpoint == "None" else training_args.checkpoint  #  None
    safetensor = None if training_args.safetensor == "None" else training_args.safetensor

    print("Setup Data")
    # Here Please fill in the corresponding dataset's root path !!!
    root_path = "path/to/dataIndex/"

    train_split_filename = find_file(root_path, 'train') #You should not modify this line
    test_split_filename = find_file(root_path, 'test') #You should not modify this line
    train_path = os.path.join(root_path, train_split_filename) #You should not modify this line
    test_path = os.path.join(root_path, test_split_filename) #You should not modify this line
    label_path = f"{root_path}label_dict.json" #You should not modify this line

    # Here Please define the train and eval_datasets using corresponding dataloader class
    train_datasets = "Please fill in the dataloader class with corresponding parameters"
    eval_datasets = "Pleased fill in the dataloader class with corresponding parameters"  

    print("Setup Model")

    # Here please define the model using the corresponding model class based on the task
    model = "Please fill in the model class with corresponding parameters"

    if safetensor is not None:
        if safetensor.endswith('.bin'):
            pretrained_weights = torch.load(safetensor)
            missing, unexpect = model.load_state_dict(pretrained_weights,strict=False)
        elif safetensor.endswith('.safetensors'):
            missing, unexpect = load_model(model, safetensor, strict=False)
        else:
            raise ValueError("Invalid safetensors!")
        print(f"Missing: {missing}")
        print(f"Unexpect: {unexpect}")
    
    print("Setup Trainer")
    trainer = Trainer(
        model=model,
        train_dataset=train_datasets,
        eval_dataset=eval_datasets,
        args=training_args,
        data_collator=DataCollator(),
        compute_metrics=compute_metrics,
    )
    # trainer.integrate_wandb()
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB__SERVICE_WAIT"] = "200"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    trainer.add_callback(metrics_callback)
    
    print("Start Training")
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_state()
    print(trainer.evaluate())

if __name__ == "__main__":
    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main()