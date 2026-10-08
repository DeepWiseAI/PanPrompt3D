# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Upstream license: licenses/Apache-2.0.txt.
# Modified for the PanPrompt3D release: packaging/import cleanup.
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Type
from torch import nn


class DSCBlock(nn.Module):

    def __init__(self, embed_dim: int, act: Type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        self.pwconv1 = nn.Conv3d(
            embed_dim, 4 * embed_dim, kernel_size=1, groups=embed_dim
        )
        self.pwconv2 = nn.Conv3d(
            4 * embed_dim, embed_dim, kernel_size=1, groups=embed_dim
        )
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 4, 1, 2, 3)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        x = x.permute(0, 2, 3, 4, 1)
        return x


class MLPBlock(nn.Module):

    def __init__(
        self, embedding_dim: int, mlp_dim: int, act: Type[nn.Module] = nn.GELU
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


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


class ImageEncoderViT3D(nn.Module):

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbed3D(
            kernel_size=(patch_size, patch_size, patch_size),
            stride=(patch_size, patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    self.img_size // patch_size,
                    self.img_size // patch_size,
                    self.img_size // patch_size,
                    embed_dim,
                )
            )
        self.triplet_attention = TripletAttention3DPost(out_chans)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block3D(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(
                    img_size // patch_size,
                    img_size // patch_size,
                    img_size // patch_size,
                ),
            )
            self.blocks.append(block)
        self.pwconv1 = nn.Conv3d(
            embed_dim, 4 * embed_dim, kernel_size=1, groups=embed_dim
        )
        self.pwconv2 = nn.Conv3d(
            4 * embed_dim, embed_dim, kernel_size=1, groups=embed_dim
        )
        self.neck = nn.Sequential(
            nn.Conv3d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm3d(out_chans),
            nn.Conv3d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm3d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x1 = self.triplet_attention(x.permute(0, 4, 1, 2, 3))
        for blk in self.blocks:
            x = blk(x)
        x = x + x1.permute(0, 2, 3, 4, 1)
        x = self.neck(x.permute(0, 4, 1, 2, 3))
        return x


class Block3D(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        which_attn: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        if which_attn == 0:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                input_size=(
                    input_size
                    if window_size == 0
                    else (window_size, window_size, window_size)
                ),
            )
        elif which_attn == 1:
            self.attn = TripletAttention3D(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                input_size=(
                    input_size
                    if window_size == 0
                    else (window_size, window_size, window_size)
                ),
            )
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(
            embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer
        )
        self.dsc = DSCBlock(embed_dim=dim, act=act_layer)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            D, H, W = (x.shape[1], x.shape[2], x.shape[3])
            x, pad_dhw = window_partition3D(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition3D(x, self.window_size, pad_dhw, (D, H, W))
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        x = x + self.dsc(self.norm2(x))
        return x


class Attention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** (-0.5)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            self.rel_pos_d = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[2] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W, _ = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, D * H * W, 3, self.num_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.reshape(3, B * self.num_heads, D * H * W, -1).unbind(0)
        attn = q * self.scale @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn,
                q,
                self.rel_pos_d,
                self.rel_pos_h,
                self.rel_pos_w,
                (D, H, W),
                (D, H, W),
            )
        attn = attn.softmax(dim=-1)
        x = (
            (attn @ v)
            .view(B, self.num_heads, D, H, W, -1)
            .permute(0, 2, 3, 4, 1, 5)
            .reshape(B, D, H, W, -1)
        )
        x = self.proj(x)
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


class TripletAttention3D(nn.Module):

    def __init__(self, dim, kernel_size=7):
        super(TripletAttention3D, self).__init__()
        self.conv1 = nn.Conv3d(
            dim,
            dim,
            kernel_size=(kernel_size, 1, 1),
            padding=(kernel_size // 2, 0, 0),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(dim)
        self.conv2 = nn.Conv3d(
            dim,
            dim,
            kernel_size=(1, kernel_size, 1),
            padding=(0, kernel_size // 2, 0),
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(dim)
        self.conv3 = nn.Conv3d(
            dim,
            dim,
            kernel_size=(1, 1, kernel_size),
            padding=(0, 0, kernel_size // 2),
            bias=False,
        )
        self.bn3 = nn.BatchNorm3d(dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = x.permute(0, 4, 1, 2, 3)
        x1 = F.adaptive_avg_pool3d(x1, (x.size(1), 1, 1))
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.sigmoid(x1)
        x1 = x1.permute(0, 2, 3, 4, 1)
        x2 = x.permute(0, 4, 1, 3, 2)
        x2 = F.adaptive_avg_pool3d(x2, (x.size(1), 1, 1))
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = self.sigmoid(x2)
        x2 = x2.permute(0, 2, 4, 3, 1)
        x3 = x.permute(0, 4, 1, 2, 3)
        x3 = F.adaptive_avg_pool3d(x3, (x.size(1), 1, 1))
        x3 = self.conv3(x3)
        x3 = self.bn3(x3)
        x3 = self.sigmoid(x3)
        x3 = x3.permute(0, 2, 3, 4, 1)
        attention = (x1 + x2 + x3) / 3
        return x * attention


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


def window_partition3D(
    x: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    B, D, H, W, C = x.shape
    pad_d = (window_size - D % window_size) % window_size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
    Hp, Wp, Dp = (H + pad_h, W + pad_w, D + pad_d)
    x = x.view(
        B,
        Dp // window_size,
        window_size,
        Hp // window_size,
        window_size,
        Wp // window_size,
        window_size,
        C,
    )
    windows = (
        x.permute(0, 1, 3, 5, 2, 4, 6, 7)
        .contiguous()
        .view(-1, window_size, window_size, window_size, C)
    )
    return (windows, (Dp, Hp, Wp))


def window_unpartition3D(
    windows: torch.Tensor,
    window_size: int,
    pad_dhw: Tuple[int, int, int],
    dhw: Tuple[int, int, int],
) -> torch.Tensor:
    Dp, Hp, Wp = pad_dhw
    D, H, W = dhw
    B = windows.shape[0] // (Dp * Hp * Wp // window_size // window_size // window_size)
    x = windows.view(
        B,
        Dp // window_size,
        Hp // window_size,
        Wp // window_size,
        window_size,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Dp, Hp, Wp, -1)
    if Hp > H or Wp > W or Dp > D:
        x = x[:, :D, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = q_coords - k_coords + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_d: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int, int],
    k_size: Tuple[int, int, int],
) -> torch.Tensor:
    q_d, q_h, q_w = q_size
    k_d, k_h, k_w = k_size
    Rd = get_rel_pos(q_d, k_d, rel_pos_d)
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_d, q_h, q_w, dim)
    rel_d = torch.einsum("bdhwc,dkc->bdhwk", r_q, Rd)
    rel_h = torch.einsum("bdhwc,hkc->bdhwk", r_q, Rh)
    rel_w = torch.einsum("bdhwc,wkc->bdhwk", r_q, Rw)
    attn = (
        attn.view(B, q_d, q_h, q_w, k_d, k_h, k_w)
        + rel_d[:, :, :, :, None, None]
        + rel_h[:, :, :, None, :, None]
        + rel_w[:, :, :, None, None, :]
    ).view(B, q_d * q_h * q_w, k_d * k_h * k_w)
    return attn


class PatchEmbed3D(nn.Module):

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16, 16),
        stride: Tuple[int, int] = (16, 16, 16),
        padding: Tuple[int, int] = (0, 0, 0),
        in_chans: int = 1,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.permute(0, 2, 3, 4, 1)
        return x
