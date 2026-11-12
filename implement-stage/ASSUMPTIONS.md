# Assumptions (ASK=never)

| ID | Gap | Chosen | Class | Source |
|---|---|---|---|---|
| A1 | Published main differs from report | Extend cloned plain U-Net; provide model-independent probability/curve manifest adapter for November model | interface | default |
| A2 | Label semantics | Explicit native-resolution tri-state boundary image: 255 visible target, 0 reviewed negative, 128 unknown. No automatic polygon closure or unmarked negatives | semantic | default |
| A3 | Missing coarse predictions | Training requires caller-exported probability arrays and ordered curves; never synthesize patient references | semantic | default |
| A4 | Output geometry | Bounded normal offsets plus reject; rejected points split fragments; no joining across gaps or temporal propagation of refined curves | interface | default |
| A5 | Leakage contract | Manifest records patient and coarse training-patient IDs; reject same-patient coarse training, train/validation patient overlap and test splits | semantic | default |
| A6 | Coarse API | Keep region mask/selected_contour for existing tracking; add refined_boundary_fragments and separate open-curve overlay/JSON | interface | default |
| A7 | Missing data package | Restore dataset/preprocessing/closed-region contour utilities required by upstream imports; these remain region semantics | interface | default |
| A8 | Sampling/loss | Equally weight patients using inverse candidate-count weights; supervised offsets from reviewed positives within local capture band; negatives only explicitly reviewed center proposals; unknown ignored | semantic | default |
| A9 | Checkpoint selection | Select lowest patient-balanced validation cross-entropy, not patient test F1; thresholds configurable, no claim of calibration | semantic | default |
| A10 | Smoke scope | Clearly labelled synthetic geometry/optimizer/checkpoint smoke only; not experimental efficacy or clinical performance | semantic | default |
