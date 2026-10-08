"""Predict with user-provided clicks; input is already resampled/cropped."""

import argparse
import json
from pathlib import Path

import torchio as tio

from panprompt3d import InteractivePredictor, build_panprompt3d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--points",
        required=True,
        help="JSON with points [[x,y,z],...] and labels [1,0,...]",
    )
    parser.add_argument("--checkpoint", default="checkpoints/panprompt3d_full_best.pth")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    points = json.loads(Path(args.points).read_text())
    coordinates, labels = points["points"], points["labels"]
    if not coordinates or len(coordinates) != len(labels):
        parser.error("Provide at least one point and one label for every point")
    if Path(args.output).exists():
        parser.error("Output already exists; choose a new filename")
    image = tio.ScalarImage(args.image)
    if tuple(image.spatial_shape) != (128, 128, 128) or image.data.shape[0] != 1:
        parser.error("Image must be a preprocessed, single-channel 128-cubed ROI")
    predictor = InteractivePredictor(build_panprompt3d(args.checkpoint), args.device)
    predictor.set_image(image.data.unsqueeze(0))
    for point, label in zip(coordinates, labels):
        mask = predictor.click(point, label)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    tio.LabelMap(tensor=mask[0], affine=image.affine).save(args.output)
    print(f"Prediction completed with {len(coordinates)} manual interaction(s)")


if __name__ == "__main__":
    main()
