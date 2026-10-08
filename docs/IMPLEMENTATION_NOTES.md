# Implementation and checkpoint notes

## Scope and provenance

Only the full `vit_b_ori` PanPrompt3D architecture is included. Its model state has
382 entries and 100,643,524 scalar values, including buffers. Layer definitions,
parameter names and numerical operations were retained from the existing project.
Unused imports, commented debugging code, and unrelated modules were removed.
`source_provenance.json` records hashes without disclosing private source locations.

The full merged checkpoint has CF loss enabled, alpha 0.5, gamma 0.5, DataAugV1,
200 configured epochs, and initialization from a prior Water model. The best saved
checkpoint carries epoch counter 192; the last checkpoint is recorded separately
in the weight manifest. "Best" was selected by training Dice, not an independent
validation set. No claim of reproducing a paper metric is made by this package.

## Historical behavior intentionally retained

- `num_clicks_max=30` is a configured parameter; the actual batch path executes 11
  interaction iterations. The sampled per-epoch click count is not used by that
  batch path. Do not describe this as 30 effective training interactions.
- Unless `--multi_click` is set, only the current point is passed; previous mask
  feedback is still used. A random middle interaction and the last training
  interaction omit points, as in the original implementation.
- Distributed training uses `seed + rank`. The inherited single-GPU path resets
  the random generators to 2023 even when `--seed` has another value.
- The original accumulation boundary is zero-based (`step % accumulation_steps
  == 0 and step != 0`); the remaining tail is not flushed. These semantics were
  not silently changed during packaging.
- The original loader reconstructs arrays via SimpleITK before TorchIO processing,
  which does not preserve all input orientation/spacing metadata. Its label-guided
  crop and recursive replacement of unsuitable training samples are retained.
- The original CF loss uses a channel-wise softmax on one-channel logits. In the
  binary one-channel path, that softmax is identically one: the retained CF term
  should not be interpreted as a newly corrected probability-based volume loss.
- The public prediction API uses the encoder/prompt-encoder/mask-decoder sequence,
  not the inherited generic `Sam3D.forward` convenience method.

These notes matter for scientific reproducibility. Packaging is not a new training
experiment and does not validate historical data splits or published metrics.

## Portability and safety changes

Real data roots are required CLI arguments. The full architecture is the only
registry choice. CF loss is enabled by default. Public checkpoints contain CPU
tensors only; no private arguments, dataset paths or optimizer payloads are exposed
in model-only files. Initialization tolerates the prior Water model's extra keys,
as did the original training code. Final-model prediction requires a strict match.
Loader workers may be zero for debugging. Single-GPU indexing uses the first
visible GPU. No source training job, dataset, or environment was modified.

The source snapshot represents the currently available implementation; an archived
training-time source commit was not available. Strict checkpoint compatibility
and code-path tests are not a guarantee of bitwise retraining equivalence.

## Verification performed

All four model class/function trees match the source after comment/docstring
cleanup. Four CPU unit tests and both CLI help entries pass. Exported tensors are
exactly equal to their sources, four checkpoint checksums pass, and serialized
checkpoint metadata passes the private-address scan. A bounded synthetic CUDA
test passes strict loading, a full encoder forward, the actual 11-interaction
training helper and decoder backward. It does not test encoder backward, an
optimizer update, multi-GPU training or a full training run. See `VERIFICATION.json`.
