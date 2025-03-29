## DTM Pretraining

### Example: Pre-training DTM no ImageNet-1K
We present an example for pre-training the DTM base model on 8 V100-32GB.

```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${num_gpus_per_node} --nnodes=${WORLD_SIZE} --node_rank=${RANK}  --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} --use_env run_pretrain_dtm.py \
        --data_set image_folder \
        --data_path ${local_data_path}/train \
        --output_dir ${SAVE_BASE_PATH}/pretrain \
        --model ${model_name}_${vocab_name}\
        --shared_lm_head True \
        --num_mask_patches 75 \
        --second_input_size 224 \
        --second_interpolation bicubic \
        --min_crop_scale ${min_crop_scale} \
        --batch_size ${batch_size} \
        --lr 1.5e-3 \
        --warmup_epochs ${pretrain_warm_up} \
        --clip_grad 3.0 \
        --drop_path ${drop_path} \
        --layer_scale_init_value 0.1 \
        --imagenet_default_mean_and_std \
        --opt_betas 0.9 ${opt_betas} \
        --opt_eps 1e-8  \
        --epochs ${epochs} \
        --print_freq 400 \
        --save_periods last every_${epochs}_epochs \
        --log_dir ${log_dir} \
        --log_name pretrain \
        --accum_iter ${pretrain_accum_iter} \
        --ln_head \
        --use_clip \
        --loss_type ${loss_type} \
        --n1 ${n1} \
        --n2 ${n2} \
        --k1 ${k1} \
        --k2 ${k2} \
        --L ${l1} \
        --auto_resume 
```
