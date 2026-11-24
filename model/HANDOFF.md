# Four-person research handoff

The completed 2026-10-22 training handoff is available in the
[study archive](experiments/cross_patient_20261022_01/README.md). It maps the actual
training file to all 24 configurations, anonymous splits, epoch histories,
per-image metrics, historical-checkpoint comparison and runnable numeric checks.
The planning checklist below still applies to new data and sequence studies.

This work is an engineering baseline, not clinical validation. The code predicts the foreground class in the supplied masks. Before training, the clinical lead and project owner must agree whether that foreground is an ICG region, an intersegmental plane, or a specifically defined boundary representation. Do not treat one as another implicitly.

## Work to complete before training

1. **Clinical/data lead:** define annotation instructions with a thoracic surgeon; curate de-identified frames and videos; decide how ambiguous, occluded, and absent boundaries are labelled. Ensure ethics/consent and local data governance approvals.
2. **Algorithm lead:** review image modality and choose `data.in_channels` / `model.in_channels` (1 grayscale or 3 RGB), set model size and training options, and sanity-check a grid of image/mask pairs and overlayed labels.
3. **Data/evaluation lead:** create train/validation/test directories split by patient or operation, preserve a private mapping to procedure IDs, and separately annotate video contours at selected timestamps—especially after fluorescence loss.
4. **Integration/experiment lead:** install the documented environment, run tests, train only after reviewed labels exist, archive config/checkpoint/logs outside source control, and maintain experiment and failure-case records.

The team has four independent responsibilities; assign the roles above to named members and agree a clinical decision-maker for annotation disputes.

## Required data and decisions

- Patient/procedure-approved ICG stills or extracted frames plus one paired binary mask per frame, with exact foreground semantics.
- Ordinary laparoscopic video with frame-level independent reference contours for temporal evaluation; record the fluorescence-disappearance time for drift analysis.
- Written component-selection policy if multiple disconnected predictions may include distractors.
- Patient/procedure identifiers held securely for leakage-free splits; never upload patient data or identifiers to GitHub.
- Camera color/order, source bit depth, frame rate, image orientation, and annotation coordinate conventions.
- A clinically chosen acceptable error/failure threshold for any future study; this baseline does not prescribe a safe-use threshold.

## Reproducible sequence

From `model/`:

```powershell
python -m pytest tests -q
python scripts/train.py --config configs/config.yaml --device cpu
python scripts/evaluate.py --config configs/config.yaml --model checkpoints/best_model.pt --split test
python scripts/predict.py --config configs/config.yaml --input path/to/video.mp4 --output predictions/video.mp4
```

For video assessment, compare saved per-frame `tracked_contour` arrays to separately reviewed contours via `Evaluator.evaluate_tracking_sequence`. Supply elapsed seconds since fluorescence loss to obtain drift-by-time bins. Report patient/procedure-level results, the number of annotations, failure cases, processing latency, and uncertainty—not just pooled pixel metrics.

## Research limitations / decisions for the next iteration

- The binary region-to-surgical-boundary relationship must be clinically demonstrated, not assumed.
- Dense Farneback flow can drift or follow the wrong texture. Its simple validity gates only detect some errors. A missing/unreliable boundary is deliberately not drawn as if trusted.
- This implementation re-runs segmentation on every video frame; segmentation confidence filtering is a prototype heuristic and requires calibration.
- Coordinate errors are in pixels, not millimeters; camera calibration is not implemented.
- No clinical dataset or independent sequence annotation is checked into this repository. Synthetic data are only for exercising software and must not appear as clinical evidence.
- The project is not cleared for intraoperative guidance, prospective clinical use, or patent freedom to operate. Obtain required regulatory, ethics, and intellectual-property review separately.
