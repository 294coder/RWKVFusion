## MFF Test
# --dataset [mff_whu, realmff, mff_mffw]

# python -u -m accelerate.commands.launch \
# --gpu_ids 0 \
# --config_file configs/huggingface/accelerate.yaml \
# runners/inferencer/accelerate_inference_on_fusion.py \
# -c "configs/RWKVFusion_config.yaml" \
# -m "RWKVFusion_MFF" \
# --val_bs 1 --dataset mff_whu \
# --dataset_mode 'test' \
# --extra_save_name 'MFF_WHU_test' \
# --model_path "log_file/pretrained/RWKVFusion_MFF.safetensors" \
# --only_y \
# --pad_window_base 32 \
# --normalize \
# --fusion_task MFF


## VIF Test
# --dataset [msrs, llvip, m3fd, roadscene, tno]

# python -u -m accelerate.commands.launch \
# --gpu_ids 1 \
# --config_file configs/huggingface/accelerate.yaml \
# runners/inferencer/accelerate_inference_on_fusion.py \
# -c "configs/RWKVFusion_config.yaml" \
# -m "RWKVFusion_VIF" \
# --val_bs 1 --dataset msrs \
# --dataset_mode 'test' \
# --extra_save_name 'VIF' \
# --model_path "log_file/pretrained/RWKVFusion_MFF.safetensors" \
# --only_y \
# --pad_window_base 32 \
# --normalize \
# --fusion_task VIF

## MEF task
# --dataset [sice, mefb, lytro]

# python -u -m accelerate.commands.launch \
# --gpu_ids 1 \
# --config_file configs/huggingface/accelerate.yaml \
# runners/inferencer/accelerate_inference_on_fusion.py \
# -c "configs/RWKVFusion_config.yaml" \
# -m "RWKVFusion_MEF" \
# --val_bs 1 --dataset sice \
# --dataset_mode 'test' \
# --extra_save_name 'MEF' \
# --model_path "log_file/RWKVFusion_v12_RWKVFusion/sice/2024-10-31-21-41-59_panRWKV_mt7rqoyn_sice_rwkv5_2 wo omnishift/weights/ema_model.pth/model.safetensors" \
# --only_y \
# --pad_window_base 32 \
# --normalize \
# --fusion_task MEF


## MIF task
# --dataset [med_harvard]

python -u -m accelerate.commands.launch \
--gpu_ids 1 \
--config_file configs/huggingface/accelerate.yaml \
runners/inferencer/accelerate_inference_on_fusion.py \
-c "configs/RWKVFusion_config.yaml" \
-m "RWKVFusion_MIF" \
--val_bs 1 --dataset med_harvard \
--dataset_mode 'test' \
--extra_save_name 'MIF' \
--model_path "log_file/RWKVFusion_v12_RWKVFusion/med_harvard/2024-11-10-17-34-26_panRWKV_cm5ic486_med_harvard_rwkv5_2_wo_omnishift_lerp_factor=0/weights/ema_model.pth/model.safetensors" \
--only_y \
--pad_window_base 32 \
--normalize \
--fusion_task MIF
