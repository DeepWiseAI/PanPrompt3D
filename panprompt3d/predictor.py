"""Sequential manual-click prediction on a preprocessed 128-cubed image."""

import torch
import torch.nn.functional as F
import torchio as tio


class InteractivePredictor:
    """Use the latest click and previous mask, as in the original inference loop.

    Points are voxel indices in the input tensor's X, Y, Z axes. No ground-truth
    mask, automatic click generation, crop oracle, or dataset is required here.
    """

    def __init__(self, model, device="cuda"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.embedding = None
        self.low_res_masks = None

    @torch.inference_mode()
    def set_image(self, image):
        if tuple(image.shape) != (1, 1, 128, 128, 128):
            raise ValueError("Expected image shape (1,1,128,128,128)")
        image = image.to(dtype=torch.float32, device=self.device)
        if not torch.isfinite(image).all() or not (image > 0).any():
            raise ValueError(
                "Image must be finite with a nonempty positive intensity mask"
            )
        normalizer = tio.ZNormalization(masking_method=lambda x: x > 0)
        image = normalizer(image.squeeze(1)).unsqueeze(1)
        self.embedding = self.model.image_encoder(image)
        self.low_res_masks = torch.zeros((1, 1, 32, 32, 32), device=self.device)

    @torch.inference_mode()
    def click(self, point, label=1):
        if self.embedding is None:
            raise RuntimeError("Call set_image before click")
        point = torch.as_tensor(point, dtype=torch.float32, device=self.device)
        if (
            point.shape != (3,)
            or not torch.isfinite(point).all()
            or ((point < 0) | (point >= 128)).any()
        ):
            raise ValueError("Point must be three voxel coordinates within [0,128)")
        if label not in (0, 1):
            raise ValueError("Label must be 1 (foreground) or 0 (background)")
        points = (point.reshape(1, 1, 3), torch.tensor([[label]], device=self.device))
        sparse, dense = self.model.prompt_encoder(
            points=points, boxes=None, masks=self.low_res_masks
        )
        self.low_res_masks, _ = self.model.mask_decoder(
            image_embeddings=self.embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        logits = F.interpolate(
            self.low_res_masks,
            size=(128, 128, 128),
            mode="trilinear",
            align_corners=False,
        )
        return (torch.sigmoid(logits) > 0.5).to(torch.uint8).cpu()
