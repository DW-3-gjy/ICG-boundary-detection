from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import torch

from icg_pipeline.refinement import BoundaryStripRefiner, RefinerConfig, decode_offsets, load_refiner, refine_curves, save_refiner
from icg_pipeline.static_boundary import corrected_region_mask, trace_candidates


def test_open_skeleton_trace_keeps_separate_fragments():
    probability = np.zeros((12, 18), dtype=np.float32)
    probability[4, 2:6] = 1
    probability[4, 8:13] = 1
    paths = trace_candidates(probability, .5, probability.shape)
    assert [x[:, 0].tolist() for x in paths] == [[2, 3, 4, 5], [8, 9, 10, 11, 12]]


def test_bilateral_forward_backward_and_position_ambiguity(tmp_path):
    cfg = RefinerConfig(radius=8, tangent_radius=3, hidden=8, variant='bilateral')
    model = BoundaryStripRefiner(cfg)
    inputs = torch.rand(3, 4, 17, 7)
    labels = torch.tensor([0, 8, 17])
    torch.nn.functional.cross_entropy(model(inputs), labels).backward()
    assert torch.isfinite(model.side_score[-1].weight.grad).all()
    assert model.side_score[-1].weight.grad.abs().sum() > 0
    save_refiner(tmp_path / 'bilateral.pt', model)
    assert torch.allclose(model(inputs), load_refiner(tmp_path / 'bilateral.pt')(inputs))
    logits = torch.zeros(2, 66)
    logits[1, 32] = 9
    _, reject, confidence = decode_offsets(logits, radius=32)
    assert reject[0] < .5 and confidence[0] < .2
    assert confidence[1] > .9


def test_region_only_moves_supported_existing_boundary():
    h, w = 80, 100
    probability = np.tile(np.clip(.5 + (np.arange(h) - 40.) / 12., 0, 1)[:, None], (1, w)).astype(np.float32)
    mask = (probability >= .5).astype(np.uint8)
    curve = np.array([[x, 40.] for x in range(20, 81, 4)], dtype=np.float32)
    shifted = curve + [0, 5]
    n = len(curve)
    estimate = {'points': shifted, 'accepted': [True] * n, 'position_mass': [.9] * n}
    cfg = RefinerConfig(radius=16)
    result = corrected_region_mask(probability, mask, [curve], estimate, cfg)
    assert mask[42, 48] and not result[42, 48] and result[46, 48]
    np.testing.assert_array_equal(result[:, :3], mask[:, :3])
    ambiguous = {**estimate, 'position_mass': [.05] * n}
    np.testing.assert_array_equal(corrected_region_mask(probability, mask, [curve], ambiguous, cfg), mask)
    no_transition = np.full_like(probability, .75)
    np.testing.assert_array_equal(corrected_region_mask(no_transition, mask, [curve], estimate, cfg), mask)


def test_refined_points_outside_pixel_centers_are_rejected(monkeypatch):
    model = BoundaryStripRefiner(RefinerConfig(radius=4))
    def forward(strips):
        scores = torch.zeros((len(strips), 10))
        scores[:, 5] = 10  # Move one pixel along the downward normal.
        return scores
    monkeypatch.setattr(model, 'forward', forward)
    result = refine_curves(model, np.zeros((20, 20, 3), np.uint8), np.zeros((20, 20), np.float32),
                           [np.array([[4, 18.75], [8, 18.75]], dtype=np.float32)])
    assert result['accepted'] == [False, False]
    assert result['fragments'] == []
