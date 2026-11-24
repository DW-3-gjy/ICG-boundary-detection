# Comparing the two ICG pipelines

For the **completed 2026-10-22 eight-condition / 24-run experiment**, use its
[archive and reproducible commands](experiments/cross_patient_20261022_01/README.md),
[dataset registration](experiments/cross_patient_20261022_01/DATASET.md), and
[results with statistical definitions](experiments/cross_patient_20261022_01/RESULTS.md).
The six-configuration commands below describe the generic runner and are not the
entry point used for that archived study. PR76 fixes software tracking/reporting;
it is not a new segmentation model or evidence of clinical temporal improvement.

This experiment compares static segmentation algorithms. The common gated optical-flow
tracker is a research prototype. Synthetic motion and brightness changes do not establish
performance during real fluorescence disappearance, surgical manipulation, or occlusion.

## What changes between runs

| Architecture | Control | Tversky | Combined |
|---|---|---|---|
| A: U-Net from scratch | foreground Dice + CE | Tversky + CE | Tversky + CE + boundary CE |
| B: pretrained ResNet-18 + U-Net decoder | same | same | same |

Each cell is an independent configuration in `configs/compare_*.yaml`. Shared settings
are inherited from `configs/comparison_base.yaml`. Nested mappings merge, except
`loss_params`, which replaces the parent mapping completely. RGB inputs are 512×288,
normalized using ImageNet statistics in both pipelines. Threshold is 0.5; neither
pipeline removes connected components from its evaluated mask.

Tversky uses **FP weight 0.7, FN weight 0.3**. This penalizes excess foreground relative
to annotations; there is no penalty for a correctly predicted large target. The combined
loss is `0.5*Tversky + 0.4*CE + 0.1*boundary_CE`. The boundary band has a two-pixel
radius at network resolution. It contributes zero for empty/full references.

## Environment and local data

Use Python with the base dependencies from `requirements.txt`. Pipeline B additionally
requires a Torchvision release compatible with the installed PyTorch; see
`requirements-resnet.txt` and the official PyTorch installation selector. Pre-cache
`ResNet18_Weights.IMAGENET1K_V1` if training offline. Missing weights cause an explicit
error; they are never silently replaced with random initialization. Inference loads
encoder weights from the trained checkpoint and does not download ImageNet weights.

Keep all patient images, masks, checkpoints, overlays, and identifying mappings local.
Use the existing Z-only source-group split and the private original-resolution tree:

```text
fullres/
  train/images/    train/masks/
  val/images/      val/masks/
  test/images/     test/masks/
```

Supply the review package's `split_manifest.json`. Each record needs `filename`, `split`,
and `source_batch`. The runner rejects groups crossing splits and mismatched image names.
No patient independence is inferred from source-batch names. Synthetic descendants retain
their parent's split. Do not mix Q/H frames into training without checking source relationships.

The target remains the supplied polygon interior. The polygon's closure at a frame edge
or occlusion may not represent an anatomical interface; review that distinction before
interpreting boundary scores.

## Run the comparison

From the repository root, activate the Python environment. Replace the example paths:

```powershell
python model/scripts/compare.py --data-root "D:/ICG/fullres" --manifest "D:/ICG/split_manifest.json" --output-root "model/logs/comparison_01"
```

This runs all six configurations, sequentially, for seeds 42, 43, 44: **18 runs**.
For a preliminary six-run check, add `--seeds 42` and use a different output root.
For a software-only short check, additionally use `--epochs 1`; those results are
not comparable with the 120-epoch protocol. Select a subset with, for example,
`--architectures unet --losses control`.

Before any training, the runner freezes fully resolved configurations, the manifest,
and SHA256 fingerprints of development images/masks. Fingerprints are rechecked before
and after training and after validation for every run; changed inputs invalidate the
run. It does not read test images.
Each run gets its own configuration, checkpoint, loss-component history, package
versions, optimizer-update count, original-resolution validation metrics, per-image CSV,
and green-reference/orange-prediction overlays. Existing frozen output roots are not
overwritten. If interrupted, keep the partial records and start a new output root.

Both architectures use 120 epochs, batch 5, AdamW, and cosine decay. Early stopping
is disabled. The encoder learning rate is 3e-5; other parameters use 3e-4. Encoder
BatchNorm statistics remain frozen. The best checkpoint is selected by validation
foreground Dice; equal scores retain the earlier checkpoint.

`validation_summary.json` reports means and variation across seeds. A loss variant
is marked promising relative to its architecture's control only if mean HD95 decreases
and mean Dice and recall each decrease by no more than 0.02. Precision and area ratio
remain visible. This is a research screening rule, not a clinical error tolerance.
An absent control or unmatched seeds leaves the decision unset.

## Frozen final test stage

After inspecting validation results and freezing the experiment, run:

```powershell
python model/scripts/compare.py --output-root "model/logs/comparison_01" --evaluate-test
```

This loads the frozen definitions and selected checkpoints without training. It checks
definition/checkpoint hashes and development-data fingerprints, then writes an exclusive test-start marker, refusing another
test invocation against the same experiment. If test evaluation fails partway through,
the marker remains: retain the evidence and diagnose the failure rather than silently
rerunning or adjusting thresholds. Never tune against the final test results.
Test inputs are fingerprinted at the start and rechecked before and after each evaluation.

Distance units are **original-image pixels**, not millimeters. Empty predictions remain
in the report. Nonfinite metric values are serialized as JSON `null`, with explicit
per-image metric names and aggregate nonfinite counts. Failure-aware means remain
nonfinite rather than silently excluding failures; finite-only means are labelled separately.

This holdout was already inspected in previous work. It is retrospective evidence,
not a fresh blinded test or a cross-patient generalization result.

## Individual training, evaluation, and prediction

The configurations also work with existing entry points. First edit only local paths
in the shared configuration to point to the preserved full-resolution split:

```powershell
python model/scripts/train.py --config model/configs/compare_unet_combined.yaml
python model/scripts/evaluate.py --config model/configs/compare_unet_combined.yaml --model model/checkpoints/comparison/unet_combined/best_model.pt --split val --output model/logs/val.json --overlays model/predictions/val
python model/scripts/predict.py --config model/configs/compare_unet_combined.yaml --input "D:/ICG/example.png"
```

Use the comparison runner for the archived, seeded protocol. Standalone checkpoint
loading restores architecture/input preprocessing from checkpoint metadata and rejects
incompatible explicit overrides. Legacy configurations keep legacy normalization,
background-inclusive Dice, and legacy tracking unless explicitly changed.

## Synthetic tracking checks

Isolate motion propagation using a reference initial mask, without any model weights:

```powershell
python model/scripts/benchmark_synthetic.py --config model/configs/compare_unet_control.yaml --split val --initialization reference --output model/logs/synthetic_reference
```

Then use `--initialization model --model <best_model.pt>` to assess the combined
segmentation/tracking behavior. The benchmark generates known translations, rotations,
smooth deformations, brightness changes, and temporary occlusions. Its metrics use
**working-image pixels** and are explicitly separate from source-resolution evaluation.
In reference mode only the first reference mask initializes the tracker; later masks
are reserved for evaluation. Runtime reported by this benchmark covers tracking only.
Tracking success uses all frames with eligible reference contours as its denominator,
including failed predictions. Contour distances are explicitly conditional on successful
annotated frames; failure frequency and annotated/evaluated counts are reported separately.
`mean_hausdorff_95_contour_distance` is the mean of those frames' contour HD95
values, not a percentile pooled across the sequence. The same metric fields are
present without successful annotated frames; undefined values serialize as `null`.

The gated tracker initializes above 0.75 mean interior probability, propagates before
considering corrections, and accepts corrections only above 0.75 probability and
0.5 temporal IoU. The 0.8 valid-flow fraction is a coarse quality gate; closed-contour
output additionally requires every sample to pass. Otherwise the full boundary is
suppressed with `failed_incomplete_contour`, preserving sample count/order instead
of deleting points and bridging missing arcs. Failed motion suppresses the tracked
boundary until reset/new clip. These are engineering defaults, not calibrated confidence.
The Python `GatedContourTracker.reset()` method provides explicit reinitialization.

Discrete images with unknown gaps must not be treated as adjacent frames. Real tracking
evaluation needs sequential images with known ordering/timing and independent reference
contours, including after fading and through tissue movement. Learned memory and learned
motion are deferred until such data are available.

## Verification

```powershell
python -m pytest model/tests -q
```

Tests use synthetic fixtures and explicit scratch ResNet construction or a mocked weight
provider; they never download pretrained weights or load clinical data. The CPU smoke
tests exercise real training, checkpoint restoration, and original-resolution inference
for both architectures. They establish software behavior, not segmentation improvement.
