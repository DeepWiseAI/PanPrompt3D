# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Upstream license: licenses/Apache-2.0.txt.
# Modified for the PanPrompt3D release: packaging/import cleanup.
import torch
from torch.nn import functional as F
from typing import List, Tuple, Type
from torch import Tensor, nn
import math


class WeightedSpatialAttention3D(torch.nn.Module):

    def __init__(self, channels=None, e_lambda=0.0001):
        super(WeightedSpatialAttention3D, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "lambda=%f)" % self.e_lambda
        return s

    @staticmethod
    def get_module_name():
        return "SimAM3D"

    def forward(self, x):
        b, c, d, h, w = x.size()
        n = d * h * w - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3, 4], keepdim=True)).pow(2)
        y = (
            x_minus_mu_square
            / (
                4
                * (
                    x_minus_mu_square.sum(dim=[2, 3, 4], keepdim=True) / n
                    + self.e_lambda
                )
            )
            + 0.5
        )
        return x * self.activation(y)


class MLPBlock3D(nn.Module):

    def __init__(
        self, embedding_dim: int, mlp_dim: int, act: Type[nn.Module] = nn.GELU
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class TwoWayTransformer3D(nn.Module):

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock3D(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=i == 0,
                )
            )
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self, image_embedding: Tensor, image_pe: Tensor, point_embedding: Tensor
    ) -> Tuple[Tensor, Tensor]:
        bs, c, x, y, z = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        queries = point_embedding
        keys = image_embedding
        for layer in self.layers:
            queries, keys = layer(
                queries=queries, keys=keys, query_pe=point_embedding, key_pe=image_pe
            )
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        return (queries, keys)


class TwoWayAttentionBlock3D(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock3D(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        return (queries, keys)


class Attention(nn.Module):

    def __init__(
        self, embedding_dim: int, num_heads: int, downsample_rate: int = 1
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


class LayerNorm3d(nn.Module):

    def __init__(self, num_channels: int, eps: float = 1e-06) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class MaskDecoder3D(nn.Module):

    def __init__(
        self,
        *,
        transformer_dim: int,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = TwoWayTransformer3D(
            depth=2, embedding_dim=self.transformer_dim, mlp_dim=2048, num_heads=8
        )
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.triplet_attention = TripletAttention3DPost(transformer_dim)
        self.weight_spatial_attention = WeightedSpatialAttention3D(
            channels=transformer_dim, e_lambda=0.0001
        )
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose3d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm3d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose3d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        return (masks, iou_pred)

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        if image_pe.shape[0] != tokens.shape[0]:
            pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        else:
            pos_src = image_pe
        b, c, x, y, z = src.shape
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : 1 + self.num_mask_tokens, :]
        src = src.transpose(1, 2).view(b, c, x, y, z)
        src_ = self.triplet_attention(src)
        src__ = src + src_
        upscaled_embedding = self.output_upscaling(src__)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, x, y, z = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, x * y * z)).view(
            b, -1, x, y, z
        )
        iou_pred = self.iou_prediction_head(iou_token_out)
        return (masks, iou_pred)


class MLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            (nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class ZPool3D(nn.Module):

    def forward(self, x):
        max_pool = torch.max(x, 1)[0].unsqueeze(1)
        avg_pool = torch.mean(x, 1).unsqueeze(1)
        return torch.cat((max_pool, avg_pool), dim=1)


class AttentionGate3D(nn.Module):

    def __init__(self, kernel_size=7):
        super(AttentionGate3D, self).__init__()
        self.compress = ZPool3D()
        self.conv = nn.Conv3d(
            2,
            1,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = self.sigmoid(x_out)
        return x * scale


class TripletAttention3DPost(nn.Module):

    def __init__(self, no_spatial=False, kernel_size=7):
        super(TripletAttention3DPost, self).__init__()
        self.cd = AttentionGate3D(kernel_size=kernel_size)
        self.ch = AttentionGate3D(kernel_size=kernel_size)
        self.cw = AttentionGate3D(kernel_size=kernel_size)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate3D(kernel_size=kernel_size)

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3, 4).contiguous()
        x_out1 = self.cd(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3, 4).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1, 4).contiguous()
        x_out2 = self.ch(x_perm2)
        x_out21 = x_out2.permute(0, 3, 1, 2, 4).contiguous()
        x_perm3 = x.permute(0, 4, 2, 3, 1).contiguous()
        x_out3 = self.cw(x_perm3)
        x_out31 = x_out3.permute(0, 4, 2, 3, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 4 * (x_out + x_out11 + x_out21 + x_out31)
        else:
            x_out = 1 / 3 * (x_out11 + x_out21 + x_out31)
        return x_out
