"""Bounded synthetic encoder-forward / 11-interaction decoder-backward test.

No patient data, optimizer update, dataset evaluation, or sustained training.
The encoder is deliberately run without backward to bound GPU memory.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from monai.losses import DiceCELoss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from panprompt3d import build_panprompt3d
from panprompt3d.losses import CF_Loss_3D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/panprompt3d_full_best.pth")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    np.random.seed(3407)
    torch.manual_seed(3407)
    torch.set_num_threads(2)
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        parser.error("This inherited CF-loss smoke requires an available CUDA GPU")
    torch.cuda.set_device(torch.device(args.device))
    model = build_panprompt3d(args.checkpoint).to(args.device)
    print("Strict full-model checkpoint loading passed", flush=True)
    with tempfile.TemporaryDirectory(prefix="panprompt3d-smoke-") as temp:
        root = Path(temp)
        (root / "imagesTr").mkdir()
        (root / "labelsTr").mkdir()
        old_argv = sys.argv
        sys.argv = [
            "train.py",
            "--train_roots",
            str(root),
            "--checkpoint",
            args.checkpoint,
            "--work_dir",
            str(root),
            "--num_workers",
            "0",
        ]
        try:
            spec = importlib.util.spec_from_file_location(
                "panprompt3d_smoke_training", ROOT / "train.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.argv = old_argv
        module.device = args.device
        image = torch.rand((1, 1, 128, 128, 128), device=args.device)
        gt = torch.zeros_like(image, dtype=torch.long)
        gt[:, :, 44:84, 44:84, 44:84] = 1
        model.train()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            embedding = model.image_encoder(image).float()
        print("Full encoder forward passed", flush=True)
        trainer = object.__new__(module.BaseTrainer)
        trainer.args = module.args
        trainer.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
        trainer.click_points, trainer.click_labels = [], []
        with torch.autocast("cuda", dtype=torch.float16):
            logits, segmentation_loss = trainer.interaction(
                model, embedding, gt, num_clicks=11
            )
            cf = CF_Loss_3D(128, beta=0.0, gamma=0.5)(logits, gt)
            loss = (segmentation_loss + 0.5 * cf).mean()
        assert torch.isfinite(logits).all() and torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in model.mask_decoder.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        print(
            json.dumps(
                {
                    "strict_weight_load": "passed",
                    "encoder_forward": "passed",
                    "interaction_iterations": 11,
                    "decoder_backward": "passed",
                    "loss_finite": True,
                    "loss": float(loss.detach()),
                    "encoder_backward_tested": False,
                    "optimizer_update_tested": False,
                    "peak_allocated_gpu_mb": round(
                        torch.cuda.max_memory_allocated() / 1024**2, 1
                    ),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
