# Implementation specification

Implement a trainable native-resolution local boundary refiner in the supplied GitHub repository. Inputs: BGR image, coarse foreground probability, ordered candidate curves, reviewed interface/nuisance/unknown labels and patient manifest. Outputs: bounded refined open fragments, checkpoint, original-coordinate prediction JSON, reproducible tests and bilingual handoff note. Base commit: ab576212bc9cbcb8ad51fb0aca295535192a039d. Clone: https://github.com/DW-3-gjy/ICG-boundary-detection.git; branch feature/native-boundary-refiner.

Success command from model/: python -m pytest tests -q; plus python scripts/refine_boundary.py smoke --output logs/refiner_smoke. No patient data, coarse weights, November ResNet implementation or measured clinical gains are supplied. Preserve the baseline region/tracking API; refined open interfaces are a separate output. Restore the missing data module necessary for baseline imports. No remote push or clinical training.
