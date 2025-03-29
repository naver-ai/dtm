## Fine-tuning on ImageNet-1K

### Example: Pre-training DTM no ImageNet-1K
We present an example for fine-tuning pre-trained models on 8 V100-32GB.
```
python -m torch.distributed.launch --nproc_per_node=${num_gpus_per_node} run_class_finetuning.py \
        --data_path ${local_data_path}/train \
        --eval_data_path ${local_data_path}/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --output_dir ${SAVE_BASE_PATH}/finetune_${finetune_epochs}eps_${finetune_warm_up}wu_${save_ft_prefix} \
        --model ${model_name} \
        --weight_decay 0.05 \
        --finetune ${SAVE_BASE_PATH}/pretrain/checkpoint-${epochs}.pth \
        --batch_size 128 \
        --lr 5e-4 \
        --update_freq 1 \
        --warmup_epochs ${finetune_warm_up} \
        --epochs ${finetune_epochs} \
        --layer_decay 0.65 \
        --drop_path 0.1 \
        --drop_block 0.1 \
        --mixup 0.8 \
        --cutmix 1.0 \
        --imagenet_default_mean_and_std \
        --dist_eval \
        --print_freq 400 \
        --save_periods last best \
        --log_dir ${log_dir} \
        --log_name finetune \
        --init_both_rpb \
        --auto_resume 
```
