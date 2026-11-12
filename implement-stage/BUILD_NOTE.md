# Build note

Start: 2026-11-12 17:55 Asia/Shanghai. Base ab576212bc9cbcb8ad51fb0aca295535192a039d.

| Rung | Check | Status |
|---|---|---|
| F0 restore baseline imports/data | existing tests | 14 passed |
| F1 bounded strip model, labels and fragment geometry | targeted geometry/gradient tests | 4 added tests passed |
| F2 patient manifest training, checkpoint and prediction CLI | synthetic train/reload/predict smoke | exit0, checkpoint/JSON/overlay written |
| F3 optional predictor integration and output serialization | regression suite | 18 tests passed; real checkpoint integration not evaluated |
| F4 bilingual handoff and assumption sweep | note + local read/diff check | handoff written; external cross-family review unavailable in bounded run |

No private images/checkpoints or November evaluation available. Actual patient performance and novelty not evaluated. Changes stay local on feature/native-boundary-refiner.

Verification: `python -m pytest model/tests -q` -> 18 passed in12.11s. `python scripts/refine_boundary.py smoke --output logs/refiner_smoke` -> exit0 after two synthetic epochs, save/reload/predict. `git diff --check` -> exit0. CPU torch2.15 installed into parent .upgrade-env with NumPy/OpenCV/SciPy/PyYAML/pytest. The environment and smoke imagery/checkpoints are not included in the patch. Upstream .gitignore ignored the missing source data package via generic data/; explicit exceptions added so restoration travels with the patch.
