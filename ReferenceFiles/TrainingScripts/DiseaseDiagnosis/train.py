import os
import sys
import random
from typing import Optional, Dict, Sequence
from dataclasses import dataclass, field
from functools import partial
import json

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from torch.nn import functional as F
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm
import wandb

import transformers
from transformers import (
    AutoModel,
    BertConfig,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
import evaluate
from safetensors.torch import load_model

# Here Please import the best matching dataset class
from Datapath.dataloader import "Here Please import the best matching dataset class"
from Model.DiagModule import RadNet

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    per_device_train_batch_size: int = field(default = 32)
    per_device_eval_batch_size: int = field(default = 32)
    output_dir: Optional[str] = field(default="path/to/M3Builder_repo/TrainPipeline/Logout")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    dataloader_drop_last: bool = field(default=True)

    start_class: int = field(default=0)
    end_class: int = field(default=4)
    backbone: str = field(default='resnet')
    level: str = field(default='articles')
    size: int = field(default=256)
    depth: int = field(default=32)
    ltype: str = field(default='BCE')
    augment: bool = field(default=True)
    dim: int = field(default=2)
    hid_dim: int = field(default=2048)
    checkpoint: Optional[str] = field(default=None)
    safetensor: Optional[str] = field(default=None)

@dataclass
class DataCollator(object):

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        images, marks, labels = tuple([instance[key] for instance in instances] for key in ('images', 'marks', 'labels'))
        images = torch.cat([_.unsqueeze(0) for _ in images],dim  = 0)
        marks = torch.cat([_.unsqueeze(0) for _ in marks],dim  = 0)
        labels = torch.cat([_.unsqueeze(0) for _ in labels],dim  = 0)
        
        return_dic = dict(
            image_x=images,
            marks=marks,
            labels=labels,
        )
        return return_dic
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_multilabel_AUC(labels, logits, epoch, name, split):
    auc_list = []
    fpr_list = []
    tpr_list = []
    scores = sigmoid(logits)
    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) < 2:
            continue
        auc = roc_auc_score(labels[:, i], scores[:, i])
        auc_list.append(auc)

        fpr, tpr, _ = roc_curve(labels[:, i], scores[:, i])
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    max_auc = np.max(auc_list)
    mean_auc = np.mean(auc_list)
    min_auc = np.min(auc_list)
    max_index = auc_list.index(max_auc)
    min_index = auc_list.index(min_auc)

    return mean_auc, max_auc, max_index, min_auc, min_index

class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.eval_epoch = 0

    def on_evaluate(self, args, state, control, **kwargs):
        self.eval_epoch = state.epoch

def eval_strategy(labels, logits, epoch, name, split):
    mAUC, max_auc, max_index, min_auc, min_index = calculate_multilabel_AUC(labels, logits, epoch, name, split)

    return {
        'mAUC': mAUC,
    }

metrics_callback = MetricsCallback()

def compute_metrics(eval_preds, level, name):
    epoch = metrics_callback.eval_epoch
    predictions = eval_preds.predictions
    label_ids = eval_preds.label_ids
    loss,logits,labels = predictions
    logits = np.clip(logits,-60,60)

    evalRes_head = eval_strategy(labels, logits, epoch, name, 'head')

    metrics = {
        "loss": np.mean(loss),
        'mAUC_head': evalRes_head['mAUC'],
    }
    return metrics
    
def main():
    set_seed()
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    start_class = training_args.start_class  
    end_class = training_args.end_class  
    num_classes = end_class - start_class
    backbone = training_args.backbone  
    level = training_args.level    
    size = training_args.size   
    depth = training_args.depth   
    ltype = training_args.ltype
    augment = training_args.augment 
    dim =training_args.dim
    hid_dim = training_args.hid_dim
    name_str = f"{backbone}_{level}_{depth}_{ltype}_{augment}"
    checkpoint = None if training_args.checkpoint == "None" else training_args.checkpoint  #  None
    safetensor = None if training_args.safetensor == "None" else training_args.safetensor
    print(name_str)
    print("Setup Data")

    root_path = "path/to/dataIndex/" #Here you should fill in the root path
    train_path = f"{root_path}train.json"
    eval_path = f"{root_path}test.json"
    label_path = f"{root_path}label_dict.json"
    

    # Here Please define the train and eval_datasets using corresponding dataloader class
    train_datasets = "Please fill in the dataloader class with corresponding parameters"
    eval_datasets = "Pleased fill in the dataloader class with corresponding parameters"  
    
    partial_compute_metrics = partial(compute_metrics, level=level, name=name_str)

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
    
    trainer = Trainer(
        model=model,
        train_dataset=train_datasets,
        eval_dataset=eval_datasets,
        args=training_args,
        data_collator=DataCollator(),
        compute_metrics=partial_compute_metrics,
    )

    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB__SERVICE_WAIT"] = "200"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    trainer.add_callback(metrics_callback)
    trainer.train(resume_from_checkpoint=checkpoint)

    trainer.save_state()
    print(trainer.evaluate())

if __name__ == "__main__":
    main()