export CUDA_VISIBLE_DEVICES="0,1"
export T_MAX="65536"
export NCCL_P2P_LEVEL="NVL"
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export OMP_NUM_THREADS="6"
export MAIN_PROCESS_PORT=29506

echo "Launching RWKVFusion HISR training script..."

# --dataset [cave, harvard, pavia, chikusei]

accelerate launch \
--config_file configs/huggingface/accelerate.yaml \
--num_processes 1 \
--gpu_ids "0" \
--main_process_port $MAIN_PROCESS_PORT \
runners/runner/accelerate_run_main.py \
--proj_name RWKVFusion \
-m 'RWKVFusion_Pavia' \
-c 'RWKVFusion_config.yaml' \
--dataset pavia \
--num_workers 6 -e 800 --train_bs 32 --val_bs 1 \
--aug_probs 0.0 0.0 --loss l1ssim --grad_accum_steps 1 \
--val_n_epoch 10 \
--checkpoint_every_n 10 \
--comment "pavia" \
--logger_on \
--ckpt_max_limit 10 \
