import os
import shutil
import json
import nibabel as nib
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import warnings
import torch
warnings.filterwarnings("ignore")

# Here Please import the best matching dataset class
from Datapath.dataloader import "Here Please import the best matching dataset class"

# You should not modify this function
def find_file(directory, tg):
    files = os.listdir(directory)
    files = [file for file in files if file.startswith(tg) and file.endswith('json')]
    assert len(files)==1
    return files[0]

# You should not modify this function
def format_nnunet_data(nnUnet_root, dataset_name, train_dataset, test_dataset):
    if os.path.exists(nnUnet_root):
        shutil.rmtree(nnUnet_root)
    
    os.makedirs(nnUnet_root)
    os.makedirs(os.path.join(nnUnet_root, 'nnUNet_preprocessed'))
    os.makedirs(os.path.join(nnUnet_root, 'nnUNet_raw'))
    task_id = 999 - len(os.listdir(os.path.join(nnUnet_root, 'nnUNet_raw')))
    os.makedirs(os.path.join(nnUnet_root, 'nnUNet_raw', 'nnUNet_raw_data'))
    os.makedirs(os.path.join(nnUnet_root, 'nnUNet_raw', 'nnUNet_raw_data', f'Task{str(task_id)}_{dataset_name}'))
    os.makedirs(os.path.join(nnUnet_root, 'nnUNet_raw', 'nnUNet_raw_data', f'Task{str(task_id)}_{dataset_name}','imagesTr'))
    os.makedirs(os.path.join(nnUnet_root, 'nnUNet_raw', 'nnUNet_raw_data', f'Task{str(task_id)}_{dataset_name}','labelsTr'))
    os.makedirs(os.path.join(nnUnet_root, 'nnUNet_raw', 'nnUNet_raw_data', f'Task{str(task_id)}_{dataset_name}','imagesTs'))
    os.makedirs(os.path.join(nnUnet_root, 'nnUNet_results'))

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    

    for idx, sample in enumerate(tqdm(train_loader)): # you must not edit this line
        if sample["images"].shape[-1] == 1 and sample["anomalies_mask"].shape[-1] == 1: # you must not edit this line
            image_tensor = sample["images"].squeeze(0) # you must not edit this line
            label_tensor = sample["anomalies_mask"].squeeze(0) # you must not edit this line
            image_tensor = torch.cat(16 * [image_tensor],dim=-1) # you must not edit this line
            label_tensor = torch.cat(16 * [label_tensor],dim=-1) # you must not edit this line
        else:# you must not edit this line
            image_tensor = sample["images"].squeeze() # you must not edit this line
            label_tensor = sample["anomalies_mask"].squeeze()# you must not edit this line
        
        assert image_tensor.shape == label_tensor.shape# you must not edit this line
        assert len(image_tensor.shape) == 3, f'{image_tensor.shape}'# you must not edit this line
        image_numpy = image_tensor.numpy()
        label_numpy = label_tensor.numpy()
        image_nii = nib.Nifti1Image(image_numpy, affine=np.eye(4))
        label_nii = nib.Nifti1Image(label_numpy, affine=np.eye(4))
        image_fname = dataset_name + str(idx) + '_0000.nii.gz'
        label_name = dataset_name + str(idx) + '.nii.gz'
        nib.save(image_nii, os.path.join(nnUnet_root, 'nnUNet_raw', 'nnUNet_raw_data', f'Task{int(task_id)}_{dataset_name}','imagesTr', image_fname))
        nib.save(label_nii, os.path.join(nnUnet_root, 'nnUNet_raw', 'nnUNet_raw_data', f'Task{int(task_id)}_{dataset_name}','labelsTr', label_name))
    
    return os.path.join(nnUnet_root, 'nnUNet_raw', 'nnUNet_raw_data', f'Task{str(task_id)}_{dataset_name}')

# You must not modify this function
def construct_dataset_json(
        modality,
        train_dataset,
        dataset_json_path
    ):
    labels = {"0":"background"}
    training = []

    imagesTr = os.path.join(dataset_json_path, 'imagesTr')
    labelsTr = os.path.join(dataset_json_path, 'labelsTr')

    for fname in os.listdir(imagesTr):
        label_fname = fname.split('_')[0] + '.nii.gz'
        image_path = os.path.join(imagesTr, label_fname)
        label_path = os.path.join(labelsTr, label_fname)
        training.append({
            "image": image_path,
            "label": label_path
        })
    
    tmp_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    label_list = None
    for sample in tmp_loader:
        label_list = sample["labels"]
        break

    for i in range(len(label_list)):
        labels[f"{i+1}"] = label_list[i][0]

    json_content = { 
        "modality": {
            "0": modality
        }, 
        "labels": labels,
        "numTraining": len(train_dataset), 
        "file_ending": ".nii.gz",
        "training": training
    }
    
    with open(os.path.join(dataset_json_path,'dataset.json'), 'w') as json_file:
        json.dump(json_content, json_file, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Pass a path to the script")
    parser.add_argument('--nnUnet_root', type=str, default='path/to/ExternalDataset/nnUnet_agent')
    args = parser.parse_args()
    nnUnet_root = args.nnUnet_root

    # Here Please fill in the corresponding dataset's root path !!!
    root_path = "path/to/dataIndex/"

    train_split_filename = find_file(root_path, 'train') # You should not modify this line
    test_split_filename = find_file(root_path, 'test') # You should not modify this line
    train_path = os.path.join(root_path, train_split_filename) # You should not modify this line
    test_path = os.path.join(root_path, test_split_filename) # You should not modify this line
    data_id = train_path.split('/')[-2] # You should not modify this line
    
    # Here Please define the train and eval_datasets using corresponding dataloader class
    train_datasets = "Please fill in the dataloader class with corresponding parameters"
    eval_datasets = "Pleased fill in the dataloader class with corresponding parameters"

    dataset_json_path = format_nnunet_data(nnUnet_root, data_id, train_datasets, eval_datasets) # You should not modify this line
    
    construct_dataset_json(
        # Here Please fill in the modality parameter with the chosen dataset's imaging modality, such as "CT", "MRI".
        modality= "Please fill in the chosen dataset's imaging modality",
        train_dataset=train_datasets,
        dataset_json_path = dataset_json_path
    )


if __name__=="__main__":
    main()