#!/bin/bash
OUTDIR="path/to/M3Builder_repo/TrainPipeline/Logout/log1"
PORT=29333
nnUnet_root="path/to/ExternalDataset/nnUnet_agent"
export nnUNet_raw_data_base="${nnUnet_root}/nnUNet_raw"
export nnUNet_preprocessed="${nnUnet_root}/nnUNet_preprocessed"
export RESULTS_FOLDER="${nnUnet_root}/nnUNet_results"

export CUDA_VISIBLE_DEVICES=4

torchrun --nproc_per_node=1 --master_port $PORT "path/to/M3Builder_repo/TrainPipeline/train.py" \
    --nnUnet_root $nnUnet_root \
2>&1 | tee "$OUTDIR/output.log"

if [ $? -eq 0 ]; then
    nnUNet_plan_and_preprocess -t 999 2>&1 | tee -a "$OUTDIR/output.log"

    if [ $? -eq 0 ]; then
        nnUNet_train 3d_fullres nnUNetTrainerV2 999 -1 --npz 2>&1 | tee -a "$OUTDIR/output.log"
    else
        exit 1
    fi
else
    exit 1
fi
