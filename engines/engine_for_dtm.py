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

from cgitb import enable
import math
import sys
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.utils as utils
import random
from bm.merge import bipartite_soft_matching, merge_wavg, merge_source

def sg_loss_ftn(output1, output2):
    output1 = F.normalize(output1, dim=-1, p=2)
    output2 = F.normalize(output2, dim=-1, p=2)
    return 2 - 2 * (output1 * output2).sum(dim=-1)

def smooth_l1_loss(pred, target, beta=2.0):
    """Smooth L1 loss on masked patches

    Args:
        pred: B x num_patches x D tensor of predict patches
        target: B x num_patches x D tensor of target patch values
        beta: Float value of L1 to L2 change point

    Return:
        loss: Masked smooth L1 loss
    """
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    loss = loss.mean(dim=-1)  # Per patch loss

    return loss

class NegCosine(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def forward(self, input, target):
        cos_sim = self.cos(input, target)
        return 1 - cos_sim.mean()

def train_one_epoch(model: torch.nn.Module, clip_model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, args=None, print_freq=10):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if 'smoothl1loss' in args.loss_type:
        loss_fn = nn.SmoothL1Loss(beta=2.0)
    elif args.loss_type == 'l1':
        loss_fn = nn.L1Loss()
    elif args.loss_type == 'l2':
        loss_fn = nn.MSELoss()
    elif args.loss_type == 'negcosine':
        loss_fn = NegCosine()
    elif args.loss_type == 'smoothl1':
        loss_fn = None
    else:
        raise ValueError("Unrecognized loss type {}".format(args.loss_type))


    for step, (batch, extra_info) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples, images, bool_masked_pos = batch
        images = images.to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)
        accum_iter = args.accum_iter

        list_num_merge = []
        list_matching_iter = []
        
        
        list_num_merge.append(random.randint(0, args.n1))
        list_num_merge.append(random.randint(0, args.n2))
        list_matching_iter.append(random.randint(1, args.k1))
        list_matching_iter.append(random.randint(1, args.k2))

        matching_r = []

        for l in range(args.L):
            if math.ceil(list_num_merge[l] / list_matching_iter[l]) * list_matching_iter[l] > args.n1:
                matching_r.append(math.floor(list_num_merge[l] / list_matching_iter[l]))
            else:
                matching_r.append(math.ceil(list_num_merge[l] / list_matching_iter[l]))

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                if args.use_clip:
                    if args.no_clip_proj:
                        target_feat = clip_model.encode_image(samples, return_all_tokens=True)
                    else:
                        target_feat = clip_model.encode_image(samples, return_all_tokens=True) @ clip_model.visual.proj
            bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)

        with torch.cuda.amp.autocast():  # enabled=False
            outputs_feat = model(samples, bool_masked_pos=bool_masked_pos, return_all_tokens=True)


            list_source = []
            for l in range(args.L):
                r = matching_r[l]
                if r == 0:
                    n, t, _ = target_feat.shape
                    source = torch.eye(t, device=target_feat.device)[None, ...].expand(n, t, t)
                else:
                    source = None
                    matching_size = None
                    iter = list_matching_iter[l]
                    x_ = target_feat.clone().detach()
                    for _ in range(iter):
                        metric = x_
                        merge, _ = bipartite_soft_matching(metric, r, True, False)
                        source = merge_source(merge, x_, source)
                        x_, matching_size = merge_wavg(merge, x_, matching_size)

                list_source.append(source)

            loss_1 = 0
            loss_2 = 0
            for l in range(args.L):
                source_final = list_source[l]
                outputs_final = torch.matmul(source_final.transpose(1, 2), torch.matmul(source_final, outputs_feat) / (source_final.sum(-1).unsqueeze(-1) + 1e-6))
                targets_final = torch.matmul(source_final.transpose(1, 2), torch.matmul(source_final, target_feat) / (source_final.sum(-1).unsqueeze(-1)+ 1e-6))

                if args.normalize_target:
                    mean = targets_final.mean(dim=-1, keepdim=True)
                    var = targets_final.var(dim=-1, keepdim=True)
                    targets_final = (targets_final - mean) / (var + 1.e-6)**.5

                if args.loss_type == 'smoothl1':
                    loss_2 = loss_2 + smooth_l1_loss(outputs_final, targets_final).mean()
                else:
                    loss_2 = loss_2 + loss_fn(outputs_final, targets_final).mean()

            loss_2 = loss_2 / args.L
            loss = loss_1 + args.w_dtm * loss_2
            loss_value = loss.item()
            if args.loss_mim:
                loss_1_value = loss_1.item()
            else:
                loss_1_value = 0
            loss_2_value = loss_2.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training at rank {utils.get_rank()}", force=True)
                sys.exit(1)

            if (step + 1) % args.accum_iter == 0:
                optimizer.zero_grad()

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(step + 1) % args.accum_iter == 0)
            loss_scale_value = loss_scaler.state_dict()["scale"]

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)
            metric_logger.update(loss_mim=loss_1_value)
            metric_logger.update(loss_tc=loss_2_value)
            metric_logger.update(loss_scale=loss_scale_value)
            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)
            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
            metric_logger.update(weight_decay=weight_decay_value)
            metric_logger.update(grad_norm=grad_norm)

            if log_writer is not None and (step + 1) % args.accum_iter == 0 and log_writer.logger_type() == 'tensorboard':
                log_writer.update(loss=loss_value, head="loss")
                log_writer.update(loss=loss_1_value, head="mim_loss")
                log_writer.update(loss=loss_2_value, head="tc_loss")
                log_writer.update(loss_scale=loss_scale_value, head="opt")
                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                log_writer.update(grad_norm=grad_norm, head="opt")

                log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

