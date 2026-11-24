# ICG boundary-perception research baseline

This directory contains a deliberately small, **experimental** baseline for segmenting an annotated foreground in ICG frames and propagating a contour through video with OpenCV dense Farneback optical flow.

For the two comparable U-Net / pretrained ResNet-18 pipelines, six loss configurations,
seeded comparison runner, and lightweight gated tracking, see [the English experiment guide](EXPERIMENTS.md).

The completed 2026-10-22 two-patient study has a separate [auditable archive and commands](experiments/cross_patient_20261022_01/README.md), [anonymous dataset register](experiments/cross_patient_20261022_01/DATASET.md), and [three-seed results](experiments/cross_patient_20261022_01/RESULTS.md). Its eight conditions/24 runs used the archived `source_snapshot/run_study.py`, not this generic six-condition runner. Only numeric records and code are published; patient images, masks, coordinates and weights remain private.

It is not a validated medical device. It does not infer a surgical margin from arbitrary fluorescent regions and cannot recover an unobservable or arbitrarily deformed boundary. A propagated contour is explicitly labelled as an estimate, and is suppressed when basic tracking checks fail.

## Pipeline and interfaces

```text
OpenCV BGR image/video
  -> configurable grayscale or RGB input preprocessing
  -> U-Net (two-class background/foreground logits)
  -> softmax/argmax full-resolution binary mask
  -> area-filtered external contours (largest qualifying component first)
  -> segmentation-supported initialization/correction
  -> forward/backward-checked Farneback propagation on ordinary video frames
  -> labelled full-resolution overlay + per-frame JSON metadata
```

The foreground class means **exactly whatever the human annotation encodes**. The included implementation does not decide whether the label is an ICG region, a visible intersegmental boundary, or another structure. Agree on that annotation target with the clinical team before training; those labels are not interchangeable.

| Module | Input → output | Responsibility |
|---|---|---|
| `icg_pipeline.data` | paired image/mask files → normalized tensor and class-index mask | Pair by filename stem, convert nonzero mask values to foreground, resize masks with nearest-neighbor interpolation, and keep training/inference normalization consistent. Supports PNG, JPEG, TIFF, and BMP; grayscale or RGB. |
| `icg_pipeline.models` | `(N,C,H,W)` image → two-class logits | Existing configurable U-Net. |
| `icg_pipeline.inference` | frame → full-resolution binary mask and contours; video → overlay and JSON | Decode logits by argmax, keep the largest area-qualified component for tracking, correct flow with accepted segmentations, and save only quality-checked propagation. |
| `icg_pipeline.tracking` | initialized `(x,y)` contour and consecutive BGR/gray frames → tracked contour and status | Dense Farneback flow sampled at contour points with forward/backward consistency, displacement, local-texture, in-frame, and valid-point-fraction checks. Failed propagation returns no contour. No Kalman filter. |
| `icg_pipeline.eval` | predictions and independent masks/contours → metrics | Image Dice/IoU/boundary metrics and sequence contour distance, HD95, failure frequency, elapsed-time drift bins, latency, and FPS. |

Image coordinates are `(x,y)`; arrays are indexed `[y,x]`. Resize is applied in training and inference, and masks are rescaled using nearest-neighbor. Video contours remain in original frame coordinates.

## Installation and configuration

From this directory, use Python 3.10+:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Edit `configs/config.yaml`. `data.in_channels` and `model.in_channels` must both be `1` (grayscale) or both be `3` (RGB). The current example uses grayscale and two-class logits. Image and mask basenames must match; each split uses `images/` and `masks/`. Relative paths in the YAML are resolved relative to the `model/` directory, independent of the current working directory. Use separate patient/procedure-level splits; do not put adjacent frames from one procedure into separate splits.

```text
data/
  train/images/<case>.(png|jpg|tif)    train/masks/<same stem>.png
  val/images/<case>.(png|jpg|tif)      val/masks/<same stem>.png
  test/images/<case>.(png|jpg|tif)     test/masks/<same stem>.png
```

Images without paired masks are rejected: the loader never invents empty labels. The included synthetic generator is only a software exercise and is **not** clinical training data.

## Commands

Run these from `model/`; scripts also accept absolute configuration/data paths.

```powershell
# Discover tests; CPU-friendly targeted suite
python -m pytest tests -q

# Optional toy images for pipeline smoke tests only (not clinical evidence)
python scripts/generate_data.py --output data_smoke --num-samples 20 --image-size 256 256

# Train on clinician-reviewed, patient/procedure-split annotations
python scripts/train.py --config configs/config.yaml --device cpu
# For a CUDA machine, omit --device cpu.

# Segment one frame
python scripts/predict.py --config configs/config.yaml --input data/test/images/example.png

# Video overlay (segmentation corrects propagation when accepted)
python scripts/predict.py --config configs/config.yaml --input path/to/video.mp4 --output predictions/video.mp4

# Segmentation-only video; no temporal tracking
python scripts/predict.py --config configs/config.yaml --input path/to/video.mp4 --no-track

# Image segmentation evaluation using independent masks
python scripts/evaluate.py --config configs/config.yaml --model checkpoints/best_model.pt --split test
```

Training checkpoints include model/optimizer/scheduler state, epoch, validation metrics, and the effective configuration. Image inference writes a mask and optional overlay and returns a result dictionary. Video inference writes a video plus JSON beside it (or under the configured output folder), with frame index, status, processing milliseconds, valid-point fraction, and forward/backward error where applicable. Statuses distinguish `segmentation_supported`, `propagated`, `uninitialized_no_reliable_segmentation`, and explicit `tracking_failed:*`.

Tracking limits and gates (`num_points`, minimum valid point count/fraction, forward/backward error, maximum displacement, minimum local texture, and Farneback parameters), contour area, minimum mean foreground probability, image size, channels, and output settings live in `configs/config.yaml`.

Propagation only outputs complete closed contours: every sampled point must pass
the quality checks. If even one point fails, the contour is suppressed instead of
deleting samples and joining the remaining points across an unsupported gap.
This conservative policy may stop display more often under partial occlusion.

## Evaluation protocol and data requirements

Image evaluation reports foreground Dice/IoU and configured boundary metrics in pixels. Video evaluation requires independent frame-level reference contours. Call `Evaluator.evaluate_tracking_sequence(results, reference_contours, elapsed_seconds=...)`; use `None` for frames without reference annotation. This reports symmetric nearest-contour distance, contour HD95, failure frequency, elapsed-since-ICG-loss drift grouped by supplied time, and latency/FPS when present. Never use tracker output as ground truth. FPS is measured by this implementation and is not a real-time guarantee.

No patient images, reviewed masks, procedure identifiers, or independently annotated clinical tracking sequences are included here. The anonymous retrospective static-segmentation results in the study archive cover only one held-out patient and do not establish clinical performance or dynamic tracking after fluorescence loss. The default synthetic generator creates random vessel-like shapes and cannot train or validate lung intersegmental-plane recognition.

## Known limitations

- A fluorescent-region mask is not inherently the anatomical or surgical boundary. Clinicians must define, annotate, and audit the target.
- Optical flow only follows image evidence at sampled points. Forward/backward checks catch some failures but cannot establish anatomical correctness; false-but-consistent flow remains possible.
- The first qualifying (area-ranked) segmented component initializes/corrects the track; multi-component clinical selection needs a clinically justified rule.
- Segmentation is run for every video frame as a possible correction; inference speed depends on hardware, frame size, and model.
- Current boundary distances are pixel-based (not physical units) and independent contours are sampled without vertex correspondence.
- Dataset pairing assumes masks share an image stem. Patient-level split enforcement depends on the supplied split directories; the code cannot infer patient identity from filenames.
- Clinical validation, prospective evaluation, risk management, operating-room integration, and freedom-to-operate/legal review remain out of scope.
