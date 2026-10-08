# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Upstream license: licenses/Apache-2.0.txt.
# Modified for the PanPrompt3D release: packaging/import cleanup.
import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple
from .image_encoder3D import ImageEncoderViT3D
from .mask_decoder3D import MaskDecoder3D
from .prompt_encoder3D import PromptEncoder3D


class Sam3D(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "L"

    def __init__(
        self,
        image_encoder: ImageEncoderViT3D,
        prompt_encoder: PromptEncoder3D,
        mask_decoder: MaskDecoder3D,
        pixel_mean: List[float] = [123.675],
        pixel_std: List[float] = [58.395],
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward(
        self, batched_input: List[Dict[str, Any]], multimask_output: bool
    ) -> List[Dict[str, torch.Tensor]]:
        input_images = torch.stack(
            [self.preprocess(x["image"]) for x in batched_input], dim=0
        )
        image_embeddings = self.image_encoder(input_images)
        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-3:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (
                self.image_encoder.img_size,
                self.image_encoder.img_size,
                self.image_encoder.img_size,
            ),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1], : input_size[2]]
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False
        )
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        d, h, w = x.shape[-3:]
        padd = self.image_encoder.img_size - d
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh, 0, padd))
        return x
