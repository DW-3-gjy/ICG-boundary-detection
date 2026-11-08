"""Scientific guardrails for visible-line supervision and patient isolation."""
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_ROOT / 'src'))
from icg_pipeline.boundary_study import (
    BoundaryReference, MixedBoundaryLoss, MixedBoundaryUNet, patient_macro,
    prediction_points, rasterize_shapes, reduced_boundary, seed_all, train_records,
)
from icg_pipeline.models.resnet_unet import ResNetUNet


class BoundaryStudyTests(unittest.TestCase):
    def test_open_line_does_not_create_region_or_closing_segment(self):
        d = {'shapes': [{'shape_type': 'linestrip', 'points': [[2, 2], [14, 2], [14, 14]]}]}
        region, boundary, has_region = rasterize_shapes(d, 20, 20)
        self.assertFalse(has_region)
        self.assertEqual(region.sum(), 0)
        self.assertEqual(boundary[8, 8], 0)
        self.assertEqual(boundary[2, 8], 1)

    def test_two_open_lines_are_not_joined(self):
        d = {'shapes': [{'shape_type': 'linestrip', 'points': p} for p in
                        [[[2, 2], [8, 2]], [[12, 14], [18, 14]]]]}
        _, boundary, _ = rasterize_shapes(d, 20, 20)
        self.assertEqual(boundary[8, 10], 0)

    def test_polygon_internal_target_and_canvas_edge_clipping(self):
        d = {'shapes': [{'shape_type': 'polygon', 'points': [[10, 10], [20, 10], [20, 20], [10, 20]]}]}
        region, boundary, has_region = rasterize_shapes(d, 20, 20)
        self.assertTrue(has_region)
        self.assertEqual(region[15, 15], 1)
        self.assertEqual(region[0, 0], 0)
        self.assertGreater(boundary.sum(), 0)

    def test_thin_boundary_survives_downsampling(self):
        boundary = np.zeros((100, 100), np.uint8)
        boundary[:, 1] = 1
        self.assertGreater(reduced_boundary(boundary, (10, 10)).sum(), 0)

    def test_open_curve_has_no_region_cross_entropy(self):
        logits = torch.randn(1, 2, 16, 16, requires_grad=True)
        boundary = torch.zeros(1, 16, 16)
        boundary[:, 8, 2:14] = 1
        loss, components = MixedBoundaryLoss()({'logits': logits, 'boundary_logits': None},
            torch.full((1, 16, 16), -999, dtype=torch.long), boundary, torch.tensor([False]))
        self.assertEqual(components['region'], 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_boundary_head_receives_finite_gradients(self):
        logits = torch.randn(1, 2, 16, 16, requires_grad=True)
        head = torch.randn(1, 16, 16, requires_grad=True)
        boundary = torch.zeros(1, 16, 16)
        boundary[:, 8, 2:14] = 1
        loss, _ = MixedBoundaryLoss()({'logits': logits, 'boundary_logits': head},
            torch.zeros(1, 16, 16, dtype=torch.long), boundary, torch.tensor([False]))
        loss.backward()
        self.assertTrue(torch.isfinite(head.grad).all())
        self.assertGreater(float(head.grad.abs().sum()), 0)

    def test_patient_macro_is_not_frame_weighted(self):
        result = patient_macro([{'patient': 'p1', 'f1': 1.}] * 100 + [{'patient': 'p2', 'f1': 0.}], 'f1')
        self.assertEqual(result['mean'], .5)

    def test_empty_prediction_distance_is_undefined(self):
        r = BoundaryReference(np.array([[1, 1], [2, 2]], np.float32), (20, 20)).score(np.zeros((0, 2)), distances=True)
        self.assertEqual(r['f1'], 0.)
        self.assertIsNone(r['hd95'])
        self.assertIsNone(r['assd'])

    def test_boundary_match_known_shift(self):
        reference = BoundaryReference(np.column_stack((np.arange(100), np.zeros(100))), (200, 200))
        shifted = reference.points + [0, 21]
        self.assertEqual(reference.score(shifted, 20)['f1'], 0.)
        self.assertEqual(reference.score(shifted, 40)['f1'], 1.)

    def test_native_points_use_original_pixel_centers(self):
        p = np.zeros((10, 10), np.float32)
        p[5, 5] = 1
        points = prediction_points(p, .5, 'boundary', (100, 100))
        np.testing.assert_allclose(points, [[54.5, 54.5]])

    def test_no_test_or_validation_rows_enter_training(self):
        rows = [dict(sample_id=str(i), planned_partition=p, planned_validation_fold=f,
                     batch=b, task_stratum=s) for i,p,f,b,s in [
            (0,'development','fold_1','1102','Z_label'),
            (1,'development','fold_2','1102','Z_label'),
            (2,'test','','1102','Z_label'),
            (3,'development','fold_2','historical_1022','Z_label'),
            (4,'development','fold_2','1102','L_visible_fluorescence')]]
        group = dict(include_old_training=False, include_l_training=False)
        self.assertEqual([r['sample_id'] for r in train_records(rows, group, 'fold_1')], ['1'])

    def test_head_creation_preserves_shared_initial_weights(self):
        torch.set_num_threads(2)
        seed_all(42)
        a = MixedBoundaryUNet(False, 100042, encoder_weights=None)
        seed_all(42)
        b = MixedBoundaryUNet(True, 100042, encoder_weights=None)
        for name, parameter in a.named_parameters():
            self.assertTrue(torch.equal(parameter, dict(b.named_parameters())[name]), name)

    def test_feature_refactor_preserves_segmentation_output(self):
        model = ResNetUNet(encoder_weights=None).eval()
        image = torch.randn(1, 3, 64, 96)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(image), model.output(model.forward_features(image))))

    def test_test_cache_access_is_blocked_in_separate_process(self):
        script = MODEL_ROOT / 'scripts/patient_boundary_study.py'
        snippet = (f"import importlib.util; from pathlib import Path; "
            f"s=importlib.util.spec_from_file_location('study', {str(script)!r}); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "m.forbid_test_access(Path.cwd()); "
            "open(Path.cwd()/'data/sealed_test/forbidden.npz','rb')")
        result = subprocess.run([sys.executable, '-c', snippet], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Sealed test cache cannot be opened', result.stderr)


if __name__ == '__main__':
    unittest.main()
