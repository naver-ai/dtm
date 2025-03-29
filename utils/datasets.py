# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import argparse
import os
import random

import torch
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from transforms import RandomResizedCropAndInterpolationWithTwoPic, RandomResizedCropAndInterpolationWithTwoPicTwoPair, \
                       RandomResizedCropAndInterpolationWithTwoPicAndCoord, _pil_interp
from timm.data import create_transform, ImageDataset 

from utils.masking_generator import MaskingGenerator
from utils.dataset_folder import ImageFolder
import torchvision

class RandomHorizontalFlip(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        p = torch.rand(1)
        if p < self.p:
            return TF.hflip(img)
        return img, p


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"



class DataAugmentationForBEiT(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        # oringinal beit data augmentation
        self.common_transform = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolationWithTwoPic(
                size=args.input_size, second_size=args.second_input_size, scale=(args.min_crop_scale, 1.0),
                interpolation=args.train_interpolation, second_interpolation=args.second_interpolation,
            ),
        ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        self.visual_token_transform = transforms.Compose([
            transforms.ToTensor(),])                                             

        self.masked_position_generator = MaskingGenerator(
            args.window_size, num_masking_patches=args.num_mask_patches,
            max_num_patches=args.max_mask_patches_per_block,
            min_num_patches=args.min_mask_patches_per_block,
        )

    def __call__(self, image):
        for_patches, for_visual_tokens = self.common_transform(image)
        return \
            self.patch_transform(for_patches), self.visual_token_transform(for_visual_tokens), \
            self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

class DataAugmentationForBEiTTwopair(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        # oringinal beit data augmentation
        self.common_transform_step1a = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4),
            # transforms.RandomHorizontalFlip(p=0.5),
        ])
        self.common_transform_step1b = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        self.common_transform_step2 = transforms.Compose([
            RandomResizedCropAndInterpolationWithTwoPicTwoPair(
                size=args.input_size, second_size=args.second_input_size, scale=(args.min_crop_scale, 1.0),
                interpolation=args.train_interpolation, second_interpolation=args.second_interpolation,
            ),
        ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        self.visual_token_transform = transforms.Compose([
            transforms.ToTensor(),])

        self.masked_position_generator = MaskingGenerator(
            args.window_size, num_masking_patches=args.num_mask_patches,
            max_num_patches=args.max_mask_patches_per_block,
            min_num_patches=args.min_mask_patches_per_block,
        )
        self.use_cross_view = args.use_cross_view

    def __call__(self, image):

        if self.use_cross_view:
            image_aug1 = self.common_transform_step1a(image)
            image_aug2 = self.common_transform_step1a(image)
            p_flip = torch.rand(1)
            if p_flip > 0.5:
                image_aug1 = TF.hflip(image_aug1)
                image_aug2 = TF.hflip(image_aug2)
            for_patches1, for_patches2, for_visual_tokens1, for_visual_tokens2 = self.common_transform_step2([image_aug1, image_aug2])

            return \
                self.patch_transform(for_patches1), self.patch_transform(for_patches2), \
                self.visual_token_transform(for_visual_tokens1), self.visual_token_transform(for_visual_tokens2), \
                self.masked_position_generator(), self.masked_position_generator()
        else:
            for_patches, for_visual_tokens = self.common_transform1b(image)

            return \
                self.patch_transform(for_patches), self.patch_transform(for_patches), \
                self.visual_token_transform(for_visual_tokens), self.visual_token_transform(for_visual_tokens), \
                self.masked_position_generator(), self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  common_transform_1st = %s,\n" % str(self.common_transform_step1a)
        repr += "  common_transform_1st = %s,\n" % str(self.common_transform_step1b)
        repr += "  common_transform_2nd = %s,\n" % str(self.common_transform_step2)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


class DataAugmentationForBEiTDoubleViews(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        # oringinal beit data augmentation
        self.common_transform = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolationWithTwoPic(
                size=args.input_size, second_size=args.second_input_size, scale=(args.min_crop_scale, 1.0),
                interpolation=args.train_interpolation, second_interpolation=args.second_interpolation,
            ),
        ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        self.visual_token_transform = transforms.Compose([
            transforms.ToTensor(),])

        self.masked_position_generator = MaskingGenerator(
            args.window_size, num_masking_patches=args.num_mask_patches,
            max_num_patches=args.max_mask_patches_per_block,
            min_num_patches=args.min_mask_patches_per_block,
        )

    def __call__(self, image):
        for_patches, for_visual_tokens = self.common_transform(image)
        for_patches2, for_visual_tokens2 = self.common_transform(image)
        return \
            self.patch_transform(for_patches), self.visual_token_transform(for_visual_tokens), \
            self.masked_position_generator(), \
            self.patch_transform(for_patches2), self.visual_token_transform(for_visual_tokens2), \
            self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


class DataAugmentationForBEiTDoubleViewsSameCrop(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        # oringinal beit data augmentation
        self.common_transform = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolationWithTwoPic(
                size=args.input_size, second_size=args.second_input_size, scale=(args.min_crop_scale, 1.0),
                interpolation=args.train_interpolation, second_interpolation=args.second_interpolation,
            ),
        ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        self.visual_token_transform = transforms.Compose([
            transforms.ToTensor(),])

        self.masked_position_generator = MaskingGenerator(
            args.window_size, num_masking_patches=args.num_mask_patches,
            max_num_patches=args.max_mask_patches_per_block,
            min_num_patches=args.min_mask_patches_per_block,
        )

    def __call__(self, image):
        for_patches, for_visual_tokens = self.common_transform(image)
        for_patches2, for_visual_tokens2 = self.common_transform(image)
        return \
            self.patch_transform(for_patches), self.visual_token_transform(for_visual_tokens), \
            self.masked_position_generator(), \
            self.patch_transform(for_patches2), self.visual_token_transform(for_visual_tokens2), \
            self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


class DataAugmentationForBEiTDoubleViewsAndIou(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        # oringinal beit data augmentation
        self.common_transform1 = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4),
            # RandomHorizontalFlip(p=0.5),
        ])
        self.common_transform2 = transforms.Compose([
            RandomResizedCropAndInterpolationWithTwoPicAndCoord(
                size=args.input_size, second_size=args.second_input_size, scale=(args.min_crop_scale, 1.0),
                interpolation=args.train_interpolation, second_interpolation=args.second_interpolation,
            ),
        ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        self.visual_token_transform = transforms.Compose([
            transforms.ToTensor(),])

        self.masked_position_generator = MaskingGenerator(
            args.window_size, num_masking_patches=args.num_mask_patches,
            max_num_patches=args.max_mask_patches_per_block,
            min_num_patches=args.min_mask_patches_per_block,
        )

        # # args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
        # xs = torch.linspace(args.patch_size[0] // 2, args.input_size - args.patch_size[0] // 2, steps=args.window_size[0])
        # ys = torch.linspace(args.patch_size[1] // 2, args.input_size - args.patch_size[1] // 2, steps=args.window_size[1])
        # grid_h, grid_w = torch.meshgrid(xs , ys, indexing='ij')
        # self.grid_h = grid_h
        # self.grid_w = grid_w
        # self.grid= torch.cat([grid_h, grid_w])
        # self.input_size = args.input_size
        # self.window_size = args.window_size

    def __call__(self, image):
        # h0 = image.size[0]
        w0 = image.size[1]
        p_flip1 = torch.rand(1)
        p_flip2 = torch.rand(1)
        image1 = self.common_transform1(image)
        image2 = self.common_transform1(image)
        if p_flip1 > 0.5:
            image1 = TF.hflip(image1)
        if p_flip2 > 0.5:
            image2 = TF.hflip(image2)
        for_patches, for_visual_tokens, i1, j1, h1, w1 = self.common_transform2(image1)
        for_patches2, for_visual_tokens2, i2, j2, h2, w2 = self.common_transform2(image2)
        if p_flip1 > 0.5:
            j1 = w0 - j1 - w1
        if p_flip2 > 0.5:
            j2 = w0 - j2 - w2
        box1 = torch.FloatTensor((i1, j1, i1 + h1, j1 + w1))
        box2 = torch.FloatTensor((i2, j2, i2 + h2, j2 + w2))
        iou = torchvision.ops.box_iou(box1.unsqueeze(0), box2.unsqueeze(0))

        # w0 = image.size[1]
        # p_flip1 = torch.rand(1)
        # p_flip2 = torch.rand(1)
        # image1 = self.common_transform1(image)
        # image2 = self.common_transform1(image)
        # if p_flip1 > 0.5:
        #     image1 = TF.hflip(image1)
        # if p_flip2 > 0.5:
        #     image2 = TF.hflip(image2)
        # for_patches, for_visual_tokens, i1, j1, h1, w1 = self.common_transform2(image1)
        # for_patches2, for_visual_tokens2, i2, j2, h2, w2 = self.common_transform2(image2)
        # if p_flip1 > 0.5:
        #     j1 = w0 - j1 - w1
        #     # w1 = w0 - w1
        #     grid_w1 = (j1 + w1 - (self.grid_w * w1) / self.input_size).flatten().unsqueeze(-1)
        # else:
        #     grid_w1 = (j1 + (self.grid_w * w1) / self.input_size).flatten().unsqueeze(-1)
        # if p_flip2 > 0.5:
        #     j2 = w0 - j2 - w2
        #     # w2 = w0 - w2
        #     grid_w2 = (j2 + w2 - (self.grid_w * w2) / self.input_size).flatten().unsqueeze(-1)
        # else:
        #     grid_w2 = (j2 + (self.grid_w * w2) / self.input_size).flatten().unsqueeze(-1)
        return \
            self.patch_transform(for_patches), self.visual_token_transform(for_visual_tokens), \
            self.masked_position_generator(), \
            self.patch_transform(for_patches2), self.visual_token_transform(for_visual_tokens2), \
            self.masked_position_generator(), iou

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  common_transform1 = %s,\n" % str(self.common_transform1)
        repr += "  common_transform2 = %s,\n" % str(self.common_transform2)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

class DataAugmentationForBEiTDoubleViewsAndCoord(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        # oringinal beit data augmentation
        self.common_transform1 = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4),
            # RandomHorizontalFlip(p=0.5),
        ])
        self.common_transform2 = transforms.Compose([
            RandomResizedCropAndInterpolationWithTwoPicAndCoord(
                size=args.input_size, second_size=args.second_input_size, scale=(args.min_crop_scale, 1.0),
                interpolation=args.train_interpolation, second_interpolation=args.second_interpolation,
            ),
        ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        self.visual_token_transform = transforms.Compose([
            transforms.ToTensor(),])

        self.masked_position_generator = MaskingGenerator(
            args.window_size, num_masking_patches=args.num_mask_patches,
            max_num_patches=args.max_mask_patches_per_block,
            min_num_patches=args.min_mask_patches_per_block,
        )

        # args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
        xs = torch.linspace(args.patch_size[0] // 2, args.input_size - args.patch_size[0] // 2, steps=args.window_size[0])
        ys = torch.linspace(args.patch_size[1] // 2, args.input_size - args.patch_size[1] // 2, steps=args.window_size[1])
        grid_h, grid_w = torch.meshgrid(xs , ys, indexing='ij')
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.grid= torch.cat([grid_h, grid_w])
        self.input_size = args.input_size
        self.window_size = args.window_size

    def __call__(self, image):
        # h0 = image.size[0]
        w0 = image.size[1]
        p_flip1 = torch.rand(1)
        p_flip2 = torch.rand(1)
        image1 = self.common_transform1(image)
        image2 = self.common_transform1(image)
        if p_flip1 > 0.5:
            image1 = TF.hflip(image1)
        if p_flip2 > 0.5:
            image2 = TF.hflip(image2)
        for_patches, for_visual_tokens, i1, j1, h1, w1 = self.common_transform2(image1)
        for_patches2, for_visual_tokens2, i2, j2, h2, w2 = self.common_transform2(image2)
        if p_flip1 > 0.5:
            j1 = w0 - j1 - w1
            # w1 = w0 - w1
            grid_w1 = (j1 + w1 - (self.grid_w * w1) / self.input_size).flatten().unsqueeze(-1)
        else:
            grid_w1 = (j1 + (self.grid_w * w1) / self.input_size).flatten().unsqueeze(-1)
        if p_flip2 > 0.5:
            j2 = w0 - j2 - w2
            # w2 = w0 - w2
            grid_w2 = (j2 + w2 - (self.grid_w * w2) / self.input_size).flatten().unsqueeze(-1)
        else:
            grid_w2 = (j2 + (self.grid_w * w2) / self.input_size).flatten().unsqueeze(-1)
        grid_h1 = (i1 + (self.grid_h * h1) / self.input_size).flatten().unsqueeze(-1)
        grid_h2 = (i2 + (self.grid_h * h2) / self.input_size).flatten().unsqueeze(-1)
        box1 = torch.FloatTensor((i1, j1, i1 + h1, j1 + w1))
        box2 = torch.FloatTensor((i2, j2, i2 + h2, j2 + w2))
        iou = torchvision.ops.box_iou(box1.unsqueeze(0), box2.unsqueeze(0))


        # grid_h1 = (i1 + (self.grid_h * h1) / self.input_size).flatten().unsqueeze(-1)
        patch_size_h1 = h1 / (2 * self.window_size[0])
        patch_size_w1 = w1 / (2 * self.window_size[1])
        patch_size_h2 = h2 / (2 * self.window_size[0])
        patch_size_w2 = w2 / (2 * self.window_size[1])
        match_matrix1 = (torch.abs(grid_h1 - grid_h2.transpose(0,1)) < patch_size_h1) * (torch.abs(grid_w1 - grid_w2.transpose(0,1)) < patch_size_w1)
        match_matrix2 = (torch.abs(grid_h2 - grid_h1.transpose(0,1)) < patch_size_h2) * (torch.abs(grid_w2 - grid_w1.transpose(0,1)) < patch_size_w2)
        # print(iou[0], match_matrix1.sum(), match_matrix2.sum())
        # print(i1, j1, h1, w1)
        # print(i2, j2, h2, w2)

        return \
            self.patch_transform(for_patches), self.visual_token_transform(for_visual_tokens), \
            self.masked_position_generator(), \
            self.patch_transform(for_patches2), self.visual_token_transform(for_visual_tokens2), \
            self.masked_position_generator(), match_matrix1, match_matrix2, iou

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  common_transform1 = %s,\n" % str(self.common_transform1)
        repr += "  common_transform2 = %s,\n" % str(self.common_transform2)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

def build_beit_pretraining_dataset(args):
    transform = DataAugmentationForBEiT(args)
    print("Data Aug = %s" % str(transform))
    
    return ImageFolder(args.data_path, transform=transform)

def build_beit_pretraining_dataset_two_pair(args):
    transform = DataAugmentationForBEiTTwopair(args)
    print("Data Aug = %s" % str(transform))

    return ImageFolder(args.data_path, transform=transform)

def build_beit_pretraining_dataset_double_views(args):
    transform = DataAugmentationForBEiTDoubleViews(args)
    print("Data Aug = %s" % str(transform))

    return ImageFolder(args.data_path, transform=transform)

def build_beit_pretraining_dataset_double_views_and_iou(args):
    transform = DataAugmentationForBEiTDoubleViewsAndIou(args)
    print("Data Aug = %s" % str(transform))

    return ImageFolder(args.data_path, transform=transform)

def build_beit_pretraining_dataset_double_views_and_coord(args):
    transform = DataAugmentationForBEiTDoubleViewsAndCoord(args)
    print("Data Aug = %s" % str(transform))

    return ImageFolder(args.data_path, transform=transform)
############################################### Dataset and Transforms for Tokenizer Training #########################################################

def build_vqkd_dataset(is_train, args):
    if is_train:
        t = []
        if args.color_jitter > 0.:
            t.append(transforms.ColorJitter(args.color_jitter, args.color_jitter, args.color_jitter))
        t.append(transforms.RandomResizedCrop(args.input_size, scale=(args.min_crop_scale, 1.0), interpolation=_pil_interp(args.train_interpolation)))
        t.append(transforms.RandomHorizontalFlip(0.5))
        t.append(transforms.ToTensor())
        transform = transforms.Compose(t)

    else:
        t = []
        if args.input_size < 384:
            args.crop_pct = 224 / 256
        else:
            args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=_pil_interp(args.train_interpolation)),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
        t.append(transforms.ToTensor())
        transform = transforms.Compose(t)
    
    print(f"{'Train' if is_train else 'Test'} Data Aug: {str(transform)}")

    if args.data_set == 'image_folder':
        if is_train:
            return ImageFolder(args.data_path, transform=transform)
        else:
            if args.eval_data_path == '':
                return ImageFolder(args.data_path, transform=transform)
            else:
                return ImageFolder(args.eval_data_path, transform=transform)

    else:
        raise NotImplementedError()


############################################### Dataset and Transforms for Ft #########################################################

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        index_file = args.image_folder_class_index_file
        dataset = ImageFolder(root, transform=transform, index_file=index_file)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
