# PanPrompt3D

Full PanPrompt3D model for interactive 3D pancreas segmentation.

This repository contains the full encoder, prompt encoder, mask decoder, training
pipeline, CF loss, and manual-click prediction API. No ablation scripts, external
validation experiments, patient data, internal server addresses, or private data
locations are included.

## Installation

Use Python 3.10 and a CUDA-compatible PyTorch installation. The tested runtime is
PyTorch 2.5.1 with CUDA 12.4. Create your own environment; this project does not
install into or depend on a shared environment.

```bash
python -m pip install -r requirements.txt
```

## Data

Supply your own preprocessed, paired NIfTI images and binary pancreas masks at
1.5-mm isotropic spacing. Each training root must have matching filenames:

```text
<TRAIN_ROOT>/
  imagesTr/case_001.nii.gz
  labelsTr/case_001.nii.gz
```

The historical training loader's coordinate conversion and label-guided crop are
retained. It does not resample raw scans or establish patient-disjoint splits.
Prepare and audit your splits independently. No real case identifiers are shipped.

## Full-model training

The supplied full-model checkpoint was trained with CF loss enabled. Its saved
configuration is in `configs/full_merged.json`; see the implementation notes before
attempting exact reproduction. The JSON is a record, not an automatically loaded
configuration; the CLI controls a new run.

```bash
# Set TRAIN_ROOT privately in your shell; never commit a real data location.
python train.py --train_roots "$TRAIN_ROOT" \
  --checkpoint checkpoints/initialization_water.pth \
  --task_name panprompt3d_full --work_dir work_dir \
  --num_epochs 200 --seed 3407 --gpu_ids 0 1 2 3 --multi_gpu \
  --batch_size 2 --accumulation_steps 20 --num_workers 6 \
  --which_data_aug v1 --num_clicks_max 30 \
  --lr 0.0008 --weight_decay 0.1 \
  --use_cf_loss --alpha_cf_loss 0.5 --gamma_cf_loss 0.5
```

Multiple source roots may follow `--train_roots`. For a single GPU, omit
`--multi_gpu` and use `--gpu_ids 0`; this is not bitwise equivalent to the historical
four-GPU run. Training data are supplied by the user, never distributed here.

`sam_model_latest.pth` is saved after each epoch. To resume a run, repeat its command
and add `--resume`, with the same work directory, task name, and data roots. Only
load trusted resume checkpoints. The released tensor-only model weights are for
initialization or prediction, not a substitute for optimizer/scheduler resume state.

## Prediction

Prepare a single-channel 128-cubed ROI with the same intensity convention and
spacing as training. Prompt coordinates are voxel indices in its X/Y/Z tensor
axes, not world coordinates. `examples/points.json` shows a synthetic prompt.
Each interaction uses the newest point and the previous low-resolution mask.

```bash
python predict.py --image "$IMAGE_ROI" --points "$POINTS_JSON" \
  --checkpoint checkpoints/panprompt3d_full_best.pth \
  --output outputs/prediction.nii.gz
```

Prediction requires no ground-truth mask. It does not crop a full scan, simulate
ground-truth clicks, reconstruct a whole-volume prediction, or run an experiment.

## Weights and checks

See `checkpoints/README.md` and `checkpoints/manifest.json` for the full-model
weights and hashes. Only full-model weights and necessary initialization are
packaged. Weights are intentionally ignored by ordinary Git; distribute approved
weights separately as release assets or explicitly configure Git LFS.

```bash
python scripts/verify_release.py
python -m unittest discover -s tests
```

See `docs/IMPLEMENTATION_NOTES.md` for retained historical behavior and validation
limits. Public release licensing and model distribution permissions must be
confirmed by the rights holder; see `PUBLICATION_CHECKLIST.md`.
