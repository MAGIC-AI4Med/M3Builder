OUTDIR="path/to/M3Builder_repo_fjh/TrainPipeline/Logout/log1"
PORT=29333

START=0
END=4
BACKBONE="resnet"
LEVEL="articles"
SIZE=256
DEPTH=32
LTYPE="MultiLabel"
AUGMENT=True
Hid_Dim=2048
CHECKPOINT="None"
SAFETENSOR="None"

WORKER=32
BATCHSIZE=16
LR=1e-5

export CUDA_VISIBLE_DEVICES=1 # Don't modify this parameter
# export NCCL_IGNORE_DISABLED_P2P=1
# export http_proxy=http://172.16.6.115:18080  
# export https_proxy=http://172.16.6.115:18080 
torchrun --nproc_per_node=1 --master_port $PORT path/to/M3Builder_repo_fjh/TrainPipeline/train.py \
    --output_dir $OUTDIR/output \
    --num_train_epochs 100 \
    --per_device_train_batch_size $BATCHSIZE \
    --per_device_eval_batch_size $BATCHSIZE \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --learning_rate $LR \
    --save_total_limit 2 \
    --save_safetensors False \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --run_name RadNet \
    --ignore_data_skip true \
    --dataloader_num_workers $WORKER \
    --remove_unused_columns False \
    --metric_for_best_model "eval_loss" \
    --load_best_model_at_end True \
    --report_to "wandb" \
    --start_class $START \
    --end_class $END \
    --backbone $BACKBONE \
    --level $LEVEL \
    --size $SIZE \
    --depth $DEPTH \
    --ltype $LTYPE \
    --augment $AUGMENT \
    --hid_dim $Hid_Dim \
    --checkpoint $CHECKPOINT \
    --safetensor $SAFETENSOR \
2>&1 | tee $OUTDIR/output.log