## Pansharpening
# --dataset [wv3, gf2, qb]

# python -u -m accelerate.commands.launch \
# --num_processes 1 \
# --gpu_ids "1" \
# --config_file configs/huggingface/accelerate.yaml \
# runners/inferencer/accelerate_inference_on_sharpening.py \
# -c configs/RWKVFusion_config.yaml \
# -m RWKVFusion_GF2 \
# --val_bs 1 \
# --dataset gf2 \
# --model_path "log_file/pretrained/RWKVFusion_GF2.safetensors" \
# --split_patch \
# # --save_mat \

## HISR
# --dataset [cave, harvard, pavia, chikusei]

python -u -m accelerate.commands.launch \
--num_processes 1 \
--gpu_ids "1" \
--config_file configs/huggingface/accelerate.yaml \
runners/inferencer/accelerate_inference_on_sharpening.py \
-c configs/RWKVFusion_config.yaml \
-m RWKVFusion_Pavia \
--val_bs 1 \
--dataset pavia \
--model_path "log_file/RWKVFusion_v12_RWKVFusion/pavia/2024-11-07-13-13-13_panRWKV_waubyiui_pavia_rwkv5_2_wo_omnishift/weights/ema_model.pth/model.safetensors" \
--split_patch
