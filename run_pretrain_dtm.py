# --------------------------------------------------------
# BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers (https://arxiv.org/abs/2208.06366)
# Github source: https://github.com/microsoft/unilm/tree/master/beitv2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Zhiliang Peng
# Based on BEiT, timm, DeiT and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import math

from pathlib import Path
import random

from timm.models import create_model

from utils.datasets import build_beit_pretraining_dataset
from utils.optim_factory import create_optimizer
from utils.utils import NativeScalerWithGradNormCount as NativeScaler
from utils.model_saver import ModelSaver
import utils.logger as logger
import utils.utils as utils

from vqkd_teacher import clip
from engines.engine_for_dtm import train_one_epoch
import models.modeling_dtm

def get_args():
    parser = argparse.ArgumentParser('BEiT pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--unscale_lr', action='store_true')
    parser.set_defaults(unscale_lr=False)


    # tokenizer settings
    parser.add_argument("--tokenizer_weight", type=str)
    parser.add_argument("--tokenizer_model", type=str, default="vqkd_encoder_base_decoder_3x768x12_clip")
    
    # Model parameters
    parser.add_argument('--model', default='beit_base_patch16_224_8k_vocab', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--shared_rel_pos_bias', action='store_true')
    parser.add_argument('--disable_shared_rel_pos_bias', action='store_false', dest='shared_rel_pos_bias')
    parser.set_defaults(shared_rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float, 
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--num_mask_patches', default=75, type=int,
                        help='number of the visual tokens/patches need be masked')
    parser.add_argument('--max_mask_patches_per_block', type=int, default=None)
    parser.add_argument('--min_mask_patches_per_block', type=int, default=16)

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size for backbone')
    parser.add_argument('--second_input_size', default=224, type=int,
                        help='images input size for discrete vae')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # cls-pretraining settings
    parser.add_argument('--early_layers', default=9, type=int, help='early_layers, default 9 for base and 21 for large')
    parser.add_argument('--head_layers', default=2, type=int, help='head_layers')
    parser.add_argument('--shared_lm_head', default=True, type=utils.bool_flag, help='head_layers')

    # Tokenizer parameters
    parser.add_argument('--codebook_size', default=8192, type=int, help='number of codebook')
    parser.add_argument('--codebook_dim', default=32, type=int, help='number of codebook')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Augmentation parameters
    parser.add_argument('--decoupling_aug', default=False, type=utils.bool_flag, help="use decoupling aug for tokenizer and vit")
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--second_interpolation', type=str, default='lanczos',
                        help='Interpolation for discrete vae (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--min_crop_scale', type=float, default=0.08, metavar='PCT',
                        help='min_crop_scale (default: 0.08)')


    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default='', type=str, help='dataset path')
    parser.add_argument('--data_set', default='image_folder',  type=str, help='dataset path')

    parser.add_argument('--imagenet_default_mean_and_std', default=False, action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='output',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')    
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # training options
    parser.add_argument('--w_sg', type=float, default=0.25, help='Self-guidance loss weight')
    parser.add_argument('--sg_start_epoch', default=10, type=int, help='print frequency')
    parser.add_argument('--sg_vis', action='store_true')
    parser.add_argument('--nl_sg_head', action='store_true')
    parser.add_argument('--sg_fc_head', action='store_true')
    parser.add_argument('--sg_smooth_l1', action='store_true')
    parser.set_defaults(sg_fc_head=True)
    parser.add_argument('--w_ssg', type=float, default=0.25, help='Self-guidance loss weight')
    parser.add_argument('--ssg_vis', action='store_true')
    parser.add_argument('--ssg_smooth_l1', action='store_true')
    parser.add_argument('--disable_sg_fc_head', action='store_false', dest='sg_fc_head')
    parser.add_argument('--disable_sg_cross_pair', action='store_true')
    parser.add_argument('--disable_sg_self_pair', action='store_true')
    parser.add_argument('--disable_ssg_cross_pair', action='store_true')
    parser.add_argument('--disable_ssg_self_pair', action='store_true')
    parser.add_argument('--thr_iou', type=float, default=0.333, help='a threshold for iou')

    parser.add_argument('--matching_last', action='store_true')
    parser.set_defaults(matching_last=True)
    parser.add_argument('--matching_r', default = 16, type=int)
    parser.add_argument('--matching_iter', default = 6, type=int)
    parser.add_argument('--w_dtm', type=float, default=1.0, help='token cluster alignment loss weight')
    parser.add_argument('--loss_2', action='store_true')

    parser.add_argument('--ln_head', action='store_true')
    parser.add_argument('--ln_gelu', action='store_true')
    parser.add_argument('--denom', action='store_true')
    parser.set_defaults(denom=True)
    parser.add_argument('--no_denom', action='store_false', dest='denom')

    parser.add_argument('--use_vis', action='store_true')
    parser.add_argument('--add_avg', action='store_true')
    parser.add_argument('--add_avg_s', action='store_true')
    parser.add_argument('--no_ce_loss_1', action='store_true')

    parser.add_argument('--use_clip', action='store_true', default=True) # use CLIP as reconstruction target
    parser.add_argument('--no_clip_proj', action='store_true', default=False)
    parser.add_argument('--target_dim', default=512, type=int, help='dimension of target features')
    parser.add_argument('--loss_type', default='smoothl1', type=str, help='type of loss')
    parser.add_argument('--loss_2_mask', action='store_true')


    parser.add_argument('--loss_mim', action='store_true')
    parser.add_argument('--L', default=2, type=int, help='L')
    parser.add_argument('--n1', default=196, type=int, help='n1')
    parser.add_argument('--n2', default=196, type=int, help='n2')
    parser.add_argument('--k1', default=14, type=int, help='k1')
    parser.add_argument('--k2', default=14, type=int, help='k2')
    parser.add_argument('--normalize_target', action='store_true')
 
    parser.add_argument('--fixed_token', action='store_true')
    parser.add_argument('--fixed_step', action='store_true')

    parser.add_argument('--n1_fixed', default=98, type=int, help='n1')
    parser.add_argument('--k1_fixed', default=7, type=int, help='k1')

    parser.add_argument('--constant_token', action='store_true')
    parser.add_argument('--constant_step', action='store_true')
    parser.add_argument('--n1_constant', default=98, type=int, help='n1')
    parser.add_argument('--k1_constant', default=7, type=int, help='k1')
# parser.set_defaults(ln_gelu=True)

    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")

    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_rel_pos_bias=args.rel_pos_bias,
        use_shared_rel_pos_bias=args.shared_rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        vocab_size=args.codebook_size,
        output_dim=args.target_dim,
    )

    return model

def get_visual_tokenizer(args):
    print(f"Creating visual tokenizer: {args.tokenizer_model}")
    model = create_model(
            args.tokenizer_model,
            pretrained=True,
            pretrained_weight=args.tokenizer_weight,
            as_tokenzer=True,
            n_code=args.codebook_size, 
            code_dim=args.codebook_dim,
        ).eval()
    return model

def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    saver = None
    if args.output_dir:
        saver = ModelSaver(checkpoint_dir=args.output_dir, target='local', periods=args.save_periods)

    model = get_model(args)
    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    # get dataset
    dataset_train = build_beit_pretraining_dataset(args)

    clip_model, _ = clip.load("ViT-B/16", device='cpu', jit=False)


    clip_model.to(device)


    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None



    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    print("Tokenizer = %s" % str(clip_model))
    total_batch_size = args.batch_size * utils.get_world_size() * args.accum_iter
    if not args.unscale_lr:
        scaled_lr = args.lr * math.sqrt(total_batch_size / 2048.0)
        args.lr = scaled_lr
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_last_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None and log_writer.logger_type() == 'tensorboard':
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        train_stats = train_one_epoch(
            model, clip_model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            args=args,
            print_freq=args.print_freq
        )

        log_stats = {**{args.log_name + '/' + f'train/{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            save_dict = utils.save_dict(args=args, model=model, model_without_ddp=model_without_ddp,
                                        optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)
            saver.save(step=epoch, num_steps=args.epochs, state=save_dict,
                       summary={'epoch': '%d/%d' % (epoch + 1, args.epochs),
                                **log_stats})

            # Logger
            if log_writer.logger_type() == 'tensorboard':
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
