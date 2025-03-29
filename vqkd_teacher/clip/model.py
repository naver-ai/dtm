from collections import OrderedDict
from typing import Tuple, Union
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import pdb
from tome.merge import bipartite_soft_matching, merge_wavg, merge_source

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x, return_all_tokens=False):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        if return_all_tokens:
            return x
        else:
            return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, return_side_out=False, return_all_tokens=False):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x
        out = []
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        if return_side_out:
            out.append(x)
        x = self.layer2(x)
        if return_side_out:
            out.append(x)
        x = self.layer3(x)
        if return_side_out:
            out.append(x)
        x = self.layer4(x)
        if return_side_out:
            out.append(x)
        x = self.attnpool(x, return_all_tokens)
        out.append(x)
        if len(out) == 1:
            return x
        else:
            return out


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # pdb.set_trace()
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, return_intermediate_out: bool = False):
        if return_intermediate_out:
            output = []
            for block in self.resblocks:
                x = block(x)
                output.append(x)
            return output

        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.width = width
        self.heads = heads
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, return_all_tokens=False, return_all_final_tokens=False, **kwargs):

        B, nc, w, h = x.shape

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        if x.shape[1] != self.positional_embedding.shape[0]:
            x = x + self.interpolate_pos_encoding(x, w, h).to(x.dtype)
        else:
            x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # if return_all_tokens:
        #     return x[:, 1:, :]

        if return_all_tokens:
            x = self.ln_post(x)
            return x[:, 1:, :]

        if return_all_final_tokens:
            return self.ln_post(x) @ self.proj

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

    def forward_kmeans(self, x: torch.Tensor, r, iters, return_all_tokens, **kwargs):

        B, nc, w, h = x.shape

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        if x.shape[1] != self.positional_embedding.shape[0]:
            x = x + self.interpolate_pos_encoding(x, w, h).to(x.dtype)
        else:
            x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = x[:, 1:, :]
        x_new = 0
        X = x.clone().detach()
        K = r
        centroids = X[:, torch.randperm(X.size(1))[:K]]
        # print(centroids.shape)
        for i in range(iters):
            # Assign each data point to the nearest centroid
            distances = torch.cdist(X, centroids)
            _, labels = torch.min(distances, dim=2)

            # Update centroids based on the mean of the assigned data points
            for k in range(K):
                mask = (labels == k).unsqueeze(2).expand_as(X) # [128, 196, 768]
                # print(mask.shape, X.shape, labels.shape)
                if mask.sum() > 0:
                    centroids[:, k] = torch.sum(X * mask, dim=1) / (mask.sum(dim=1) + 1e-6) # 128, 768
                    # print(centroids[:, k].shape)

                if i == iters - 1:
                    x_new = x_new + mask * (torch.sum(x * mask, dim=1) / (mask.sum(dim=1) + 1e-6)).unsqueeze(1)

        if return_all_tokens:
            x = self.ln_post(x)
            x_new = self.ln_post(x_new)
            return x, x_new, labels

        else:
            x = x_new
            x = self.ln_post(x)
            return x, labels

    def forward_kmedoids(self, x: torch.Tensor, r, return_all_tokens, **kwargs):

        B, nc, w, h = x.shape

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        if x.shape[1] != self.positional_embedding.shape[0]:
            x = x + self.interpolate_pos_encoding(x, w, h).to(x.dtype)
        else:
            x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x = x[:, 1:, :]
        # x_new = 0
        # X = x.clone().detach()
        # K = r
        # centroids = X[:, torch.randperm(X.size(1))[:K]]
        # # print(centroids.shape)
        # for i in range(iters):
        #     # Assign each data point to the nearest centroid
        #     distances = torch.cdist(X, centroids)
        #     _, labels = torch.min(distances, dim=2)
        #
        #     # Update centroids based on the mean of the assigned data points
        #     for k in range(K):
        #         mask = (labels == k).unsqueeze(2).expand_as(X) # [128, 196, 768]
        #         # print(mask.shape, X.shape, labels.shape)
        #         if mask.sum() > 0:
        #             centroids[:, k] = torch.sum(X * mask, dim=1) / (mask.sum(dim=1) + 1e-6) # 128, 768
        #             # print(centroids[:, k].shape)
        #
        #         if i == iters - 1:
        #             x_new = x_new + mask * (torch.sum(x * mask, dim=1) / (mask.sum(dim=1) + 1e-6)).unsqueeze(1)

        x = x[:, 1:, :]
        x_new = 0
        X = x.clone().detach()
        k = r

        similarity_matrix = torch.cdist(X, X)
        # Step 1: Select initial medoids
        num = len(similarity_matrix)
        row_sums = torch.sum(similarity_matrix, dim=1)
        print(similarity_matrix.shape, num, row_sums.shape)
        normalized_sim = similarity_matrix.T / row_sums
        normalized_sim = normalized_sim.T
        priority_scores = -torch.sum(normalized_sim, dim=0)
        values, indices = priority_scores.topk(k)

        tmp = -similarity_matrix[:, indices]
        tmp_values, tmp_indices = tmp.topk(1, dim=1)
        min_distance = -torch.sum(tmp_values)
        cluster_assignment = tmp_indices.resize_(num)
        print(min_distance)

        # Step 2: Update medoids
        for i in range(k):
            sub_indices = (cluster_assignment == i).nonzero()
            sub_num = len(sub_indices)
            sub_indices = sub_indices.resize_(sub_num)
            sub_similarity_matrix = torch.index_select(similarity_matrix, 0, sub_indices)
            sub_similarity_matrix = torch.index_select(sub_similarity_matrix, 1, sub_indices)
            sub_row_sums = torch.sum(sub_similarity_matrix, dim=1)
            sub_medoid_index = torch.argmin(sub_row_sums)
            # update the cluster medoid index
            indices[i] = sub_indices[sub_medoid_index]

        # Step 3: Assign objects to medoids
        tmp = -similarity_matrix[:, indices]
        tmp_values, tmp_indices = tmp.topk(1, dim=1)
        total_distance = -torch.sum(tmp_values)
        cluster_assignment = tmp_indices.resize_(num)
        print(total_distance)

        while (total_distance < min_distance):
            min_distance = total_distance
            # Step 2: Update medoids
            for i in range(k):
                sub_indices = (cluster_assignment == i).nonzero()
                sub_num = len(sub_indices)
                sub_indices = sub_indices.resize_(sub_num)
                sub_similarity_matrix = torch.index_select(similarity_matrix, 0, sub_indices)
                sub_similarity_matrix = torch.index_select(sub_similarity_matrix, 1, sub_indices)
                sub_row_sums = torch.sum(sub_similarity_matrix, dim=1)
                sub_medoid_index = torch.argmin(sub_row_sums)
                # update the cluster medoid index
                indices[i] = sub_indices[sub_medoid_index]

            # Step 3: Assign objects to medoids
            tmp = -similarity_matrix[:, indices]
            tmp_values, tmp_indices = tmp.topk(1, dim=1)
            total_distance = -torch.sum(tmp_values)
            cluster_assignment = tmp_indices.resize_(num)

        if return_all_tokens:
            x = self.ln_post(x)
            x_new = self.ln_post(x_new)
            return x, x_new, indices

        else:
            x = x_new
            x = self.ln_post(x)
            return x, indices

    def forward_dtm(self, x: torch.Tensor, r, iters, **kwargs):

        B, nc, w, h = x.shape

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        if x.shape[1] != self.positional_embedding.shape[0]:
            x = x + self.interpolate_pos_encoding(x, w, h).to(x.dtype)
        else:
            x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = x[:, 1:, :]

        target_feat = x[:, 1:, :].clone().detach()

        if r == 0:
            n, t, _ = target_feat.shape
            source = torch.eye(t, device=target_feat.device)[None, ...].expand(n, t, t)

        else:

            source = None
            matching_size = None
            x_ = target_feat
            for _ in range(iters):
                metric = x_

                # Apply ToMe here
                merge, _ = bipartite_soft_matching(metric, r, True, False)
                source = merge_source(merge, x_, source)
                x_, matching_size = merge_wavg(merge, x_, matching_size)


        x_final = torch.matmul(source.transpose(1, 2), torch.matmul(source, x[:, 1:, :]) / (source.sum(-1).unsqueeze(-1)+ 1e-6))

        x = self.ln_post(x)
        x_final = self.ln_post(x_final)


        # for i, idx in enumerate(K):
        #     mask = (index_map == idx).unsqueeze(1).expand_as(features)
        #     grouped_features = features[mask].view(-1, features.size(1))
        #     summed_features = torch.mean(grouped_features, dim=0)
        #     x = x + mask * (summed_features - x)

        # if return_all_tokens:
        #     x = self.ln_post(x)
        #     return x[:, 1:, :]
        #
        # if return_all_final_tokens:
        #     return self.ln_post(x) @ self.proj
        #
        # x = self.ln_post(x[:, 0, :])

        # if self.proj is not None:
        #     x = x @ self.proj

        return x, x_final, source

    def interpolate_pos_encoding(self, x, w, h):
        # pdb.set_trace()
        npatch = x.shape[1] - 1
        N = self.positional_embedding.shape[0] - 1 # 256 for large
        if npatch == N and w == h:
            return self.positional_embedding
        class_pos_embed = self.positional_embedding[[0]]
        patch_pos_embed = self.positional_embedding[1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int, # 512
                 # vision
                 image_resolution: int, # 224
                 vision_layers: Union[Tuple[int, int, int, int], int], # 12
                 vision_width: int, # 768
                 vision_patch_size: int, # 16
                 # text
                 context_length: int, # 77
                 vocab_size: int, # 49408
                 transformer_width: int, # 512
                 transformer_heads: int, # 8
                 transformer_layers: int # 12
                 ):
        super().__init__()
        # pdb.set_trace()
        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, return_side_out=False, return_all_tokens=False, return_all_final_tokens=False, **kwargs):
        return self.visual(image.type(self.dtype), return_all_tokens, return_all_final_tokens, **kwargs)

    def encode_image_kmeans(self, image, r, iters, return_all_tokens=False, **kwargs):
        return self.visual.forward_kmeans(image.type(self.dtype), r, iters, return_all_tokens, **kwargs)

    def encode_image_kmedoids(self, image, r, return_all_tokens=False, **kwargs):
        return self.visual.forward_kmedoids(image.type(self.dtype), r, return_all_tokens, **kwargs)

    def encode_image_dtm(self, image, r, iters, **kwargs):
        return self.visual.forward_dtm(image.type(self.dtype), r, iters, **kwargs)

    def encode_text(self, text, return_all_tokens=False, return_patch_tokens=False):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        if return_patch_tokens:
            return x
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if return_all_tokens:
            # pdb.set_trace()
            x = x @ self.text_projection
        else:
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
