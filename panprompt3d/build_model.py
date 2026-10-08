# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Upstream license: licenses/Apache-2.0.txt.
# Modified: only the existing full PanPrompt3D configuration is exposed;
# public checkpoints use tensor-only, strict loading.
"""Construct the original full 128-cubed PanPrompt3D architecture."""

from functools import partial

import torch

from .modeling import ImageEncoderViT3D, MaskDecoder3D, PromptEncoder3D, Sam3D


def load_weights(model, checkpoint, *, allow_partial=False):
    """Load safe tensor dictionaries; partial loading is for initialization only."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, dict) or not all(
        torch.is_tensor(v) for v in state.values()
    ):
        raise ValueError("Expected a tensor-only model state dictionary")
    result = model.load_state_dict(state, strict=not allow_partial)
    return result


def build_panprompt3d(checkpoint=None):
    """Retain all full-model layers and the original checkpoint key names."""
    model = Sam3D(
        image_encoder=ImageEncoderViT3D(
            depth=12,
            embed_dim=768,
            img_size=128,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=384,
        ),
        prompt_encoder=PromptEncoder3D(
            embed_dim=384,
            image_embedding_size=(8, 8, 8),
            input_image_size=(128, 128, 128),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder3D(
            num_multimask_outputs=3,
            transformer_dim=384,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    if checkpoint is not None:
        load_weights(model, checkpoint)
    return model.eval()


sam_model_registry3D = {"vit_b_ori": build_panprompt3d}
